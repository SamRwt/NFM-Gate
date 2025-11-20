# nfm_cifar_train.py
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
# ------------------------
# Config
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
print("Device:", device)
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

IN_CHANNELS = 3
N_BLOCKS = 12        # more blocks -> more expressive flow
HIDDEN_DIM = 512     # larger coupling nets
BATCH_SIZE = 256
EPOCHS = 100         # longer training
LR = 1e-4
NUM_WORKERS = 32

# ------------------------
# Layers
# ------------------------
class ReverseChannels(nn.Module):
    def forward(self, x, reverse=False):
        y = x.flip(dims=[1])
        # return a zero logdet per-sample
        return y, x.new_zeros(x.size(0))

class ActNorm(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.initialized = False
        self.eps = eps

    def initialize(self, x):
        with torch.no_grad():
            mean = x.mean(dim=(0,2,3), keepdim=True)
            std = x.std(dim=(0,2,3), keepdim=True)
            # avoid zeros
            std = std + self.eps
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1.0 / std)
            self.initialized = True

    def forward(self, x, reverse=False):
        if not self.initialized:
            self.initialize(x)
        B, C, H, W = x.shape
        if not reverse:
            y = (x + self.loc) * self.scale
            logdet = torch.sum(torch.log(torch.abs(self.scale))) * H * W
        else:
            y = x / (self.scale + self.eps) - self.loc
            logdet = -torch.sum(torch.log(torch.abs(self.scale))) * H * W
        return y, logdet

class AffineCoupling(nn.Module):
    def __init__(self, in_channels, hidden=256):
        super().__init__()
        self.C = in_channels
        # deterministic split: x1 has the larger half for odd channels
        self.split = in_channels - (in_channels // 2)
        self.net = nn.Sequential(
            nn.Conv2d(self.split, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        out_ch = (in_channels - self.split) * 2
        self.st = nn.Conv2d(hidden, out_ch, 3, padding=1)
        nn.init.zeros_(self.st.weight)
        nn.init.zeros_(self.st.bias)

    def forward(self, x, reverse=False):
        x1 = x[:, :self.split, :, :]
        x2 = x[:, self.split:, :, :]
        h = self.net(x1)
        st = self.st(h)
        s, t = st.chunk(2, dim=1)
        s = torch.tanh(s)
        if not reverse:
            y2 = x2 * torch.exp(s) + t
            logdet = s.sum(dim=(1,2,3))
        else:
            y2 = (x2 - t) * torch.exp(-s)
            logdet = -s.sum(dim=(1,2,3))
        y = torch.cat([x1, y2], dim=1)
        return y, logdet

    def inverse(self, y):
        return self.forward(y, reverse=True)

# ------------------------
# Flow model
# ------------------------
class SimpleFlow(nn.Module):
    def __init__(self, in_channels=3, n_blocks=8, hidden=256):
        super().__init__()
        layers = []
        C = in_channels
        for _ in range(n_blocks):
            layers.append(ActNorm(C))
            layers.append(AffineCoupling(C, hidden=hidden))
            layers.append(ReverseChannels())
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        logdet_total = x.new_zeros(x.size(0))
        y = x
        for layer in self.layers:
            y, logdet = layer(y, reverse=False)
            logdet_total = logdet_total + logdet
        z = y
        return z, logdet_total

    def inverse(self, z):
        y = z
        for layer in reversed(self.layers):
            y, _ = layer(y, reverse=True)
        return y

    def log_prob(self, x):
        z, logdet = self.forward(x)
        z_flat = z.view(z.size(0), -1)
        D = z_flat.size(1)
        # standard normal prior
        logpz = -0.5 * (z_flat**2).sum(1) - 0.5 * D * math.log(2 * math.pi)
        return logpz + logdet

# ------------------------
# Data: CIFAR-10 (standard)
# ------------------------
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

train_ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
test_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

# ------------------------
# Training loop (MLE)
# ------------------------
def train_flow(flow, loader, epochs=EPOCHS, lr=LR, checkpoint_dir=CHECKPOINT_DIR):
    flow.to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=1e-6)
    best_nll = float('inf')
    for ep in range(1, epochs+1):
        flow.train()
        total_loss = 0.0
        n = 0
        pbar = tqdm(loader, desc=f"Epoch {ep}/{epochs}", leave=False)
        for xb, _ in pbar:
            xb = xb.to(device)
            logpx = flow.log_prob(xb)       # per-sample log p(x)
            loss = -logpx.mean()            # average NLL
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        epoch_nll = total_loss / n
        print(f"[Flow] Epoch {ep}/{epochs}  NLL: {epoch_nll:.6f}")

        # save checkpoint each epoch
        ckpt_path = os.path.join(checkpoint_dir, f"flow_epoch_{ep:03d}.pth")
        torch.save({'epoch': ep, 'model_state': flow.state_dict(), 'optimizer_state': opt.state_dict(), 'nll': epoch_nll}, ckpt_path)

        # keep best copy
        if epoch_nll < best_nll:
            best_nll = epoch_nll
            best_path = os.path.join(checkpoint_dir, "flow_best.pth")
            torch.save({'epoch': ep, 'model_state': flow.state_dict(), 'optimizer_state': opt.state_dict(), 'nll': epoch_nll}, best_path)

    # final save
    final_path = os.path.join(checkpoint_dir, "flow_final.pth")
    torch.save({'epoch': epochs, 'model_state': flow.state_dict(), 'optimizer_state': opt.state_dict(), 'nll': epoch_nll}, final_path)
    print(f"Training finished. Final model saved to {final_path}")

import matplotlib.pyplot as plt
import numpy as np
##Visualize script
def visualize_latents_and_reconstructions(flow, dataloader, n=8, save_path="latents_visualization.png"):
    """
    For n images from dataloader: visualize Original | Latent (visualized) | Reconstruction
    Saves a single image to save_path.
    """
    flow.eval()
    device_local = next(flow.parameters()).device
    xb, _ = next(iter(dataloader))
    xb = xb[:n].to(device_local)

    with torch.no_grad():
        z, _ = flow.forward(xb)       # shape (n, C, H, W)
        x_rec = flow.inverse(z)       # should be (n, C, H, W)

    # helper: convert tensor (C,H,W) to numpy image in [0,1] for plotting
    def tensor_to_image(t):
        t = t.detach().cpu().numpy()
        t = np.transpose(t, (1, 2, 0))  # H,W,C
        return np.clip(t, 0.0, 1.0)

    # helper to visualize latent z as RGB image
    def latent_to_rgb(z_tensor):
        # z_tensor: (C,H,W) torch tensor (can be any real values)
        z_np = z_tensor.detach().cpu().numpy()
        C, H, W = z_np.shape
        if C == 3:
            rgb = np.transpose(z_np, (1,2,0))  # H,W,3
        else:
            # pick top-3 channels by std
            flat = z_np.reshape(C, -1)
            stds = flat.std(axis=1)
            top3 = np.argsort(stds)[-3:]
            rgb = np.stack([z_np[ch] for ch in top3], axis=-1)  # H,W,3
        # robust min-max per-image to avoid outliers (2nd/98th percentiles)
        low = np.percentile(rgb, 2)
        high = np.percentile(rgb, 98)
        if high - low < 1e-6:
            out = np.zeros_like(rgb)
        else:
            out = (rgb - low) / (high - low)
        out = np.clip(out, 0.0, 1.0)
        return out

    # build plot grid: n rows, 3 columns
    fig, axes = plt.subplots(n, 3, figsize=(9, 3*n))
    for i in range(n):
        # original
        img_orig = tensor_to_image(xb[i])
        axes[i,0].imshow(img_orig)
        axes[i,0].axis('off')
        if i == 0:
            axes[i,0].set_title('Original')

        # latent visualization
        latent_img = latent_to_rgb(z[i])
        axes[i,1].imshow(latent_img)
        axes[i,1].axis('off')
        if i == 0:
            axes[i,1].set_title('Latent (z) visualized')

        # reconstruction
        img_rec = tensor_to_image(x_rec[i])
        axes[i,2].imshow(img_rec)
        axes[i,2].axis('off')
        if i == 0:
            axes[i,2].set_title('Reconstruction F^{-1}(z)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved latent visualization to {save_path}")
    plt.close(fig)

# ------------------------
# Entrypoint
# ------------------------
if __name__ == "__main__":
    flow = SimpleFlow(in_channels=IN_CHANNELS, n_blocks=N_BLOCKS, hidden=HIDDEN_DIM).to(device)
    print(f"Model: SimpleFlow(in_channels={IN_CHANNELS}, n_blocks={N_BLOCKS}, hidden={HIDDEN_DIM})")
    print(f"Training {EPOCHS} epochs, batch_size={BATCH_SIZE}, lr={LR}")
    # train_flow(flow, train_loader, epochs=EPOCHS, lr=LR, checkpoint_dir=CHECKPOINT_DIR)
    # Visualize latents and reconstructions for 8 images (new)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "flow_best.pth")
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            state = ckpt.get('model_state', ckpt)
            try:
                flow.load_state_dict(state)
            except Exception:
                flow.load_state_dict(state, strict=False)
            print(f"Loaded checkpoint '{ckpt_path}' (epoch={ckpt.get('epoch', '?')}, nll={ckpt.get('nll', '?')})")
        except Exception as e:
            print(f"Failed to load checkpoint '{ckpt_path}': {e}")
    else:
        print(f"No checkpoint found at '{ckpt_path}', starting from scratch.")
    visualize_latents_and_reconstructions(flow, test_loader, n=8, save_path="latents_visualization.png")

