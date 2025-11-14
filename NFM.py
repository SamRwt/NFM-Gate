import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ------------------------
# helper blocks
# ------------------------
class AffineCoupling(nn.Module):
    """
    Fix: explicit split so conv input channels match x1 shape for both even/odd C.
    """
    def __init__(self, in_channels, hidden=256):
        super().__init__()
        self.C = in_channels
        # choose x1 to be the larger half so it matches typical chunk behavior
        self.split = in_channels - (in_channels // 2)  # larger half
        self.h = nn.Sequential(
            nn.Conv2d(self.split, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU()
        )
        # output scale and translate for the other part (size = C - self.split)
        out_channels = (in_channels - self.split) * 2
        self.st = nn.Conv2d(hidden, out_channels, 3, padding=1)
        nn.init.zeros_(self.st.weight)
        nn.init.zeros_(self.st.bias)

    def forward(self, x, reverse=False):
        # explicit split to avoid torch.chunk ambiguity
        x1 = x[:, :self.split, :, :]
        x2 = x[:, self.split:, :, :]
        h = self.h(x1)
        st = self.st(h)
        s, t = st.chunk(2, 1)  # split scale/shift along channel dim
        s = torch.tanh(s)      # bounded scale
        if not reverse:
            y2 = x2 * torch.exp(s) + t
            logdet = s.flatten(1).sum(-1)
        else:
            y2 = (x2 - t) * torch.exp(-s)
            logdet = -s.flatten(1).sum(-1)
        y = torch.cat([x1, y2], dim=1)
        return y, logdet

    def inverse(self, y):
        return self.forward(y, reverse=True)


class ActNorm(nn.Module):
    """Simple channel-wise affine with data-dependent init."""
    def __init__(self, channels):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.initialized = False

    def initialize(self, x):
        # initialize using first batch
        with torch.no_grad():
            mean = x.mean(dim=(0,2,3), keepdim=True)
            std = x.std(dim=(0,2,3), keepdim=True)
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1/(std + 1e-6))
            self.initialized = True

    def forward(self, x, reverse=False):
        if not self.initialized:
            self.initialize(x)
        if not reverse:
            y = (x + self.loc) * self.scale
            # logdet: sum over channels * H * W
            _, C, H, W = x.shape
            logdet = torch.sum(torch.log(torch.abs(self.scale))) * H * W
        else:
            y = x / (self.scale + 1e-6) - self.loc
            _, C, H, W = x.shape
            logdet = -torch.sum(torch.log(torch.abs(self.scale))) * H * W
        return y, logdet


# ------------------------
# Flow model wrapper
# ------------------------
class SimpleFlow(nn.Module):
    def __init__(self, in_channels=3, n_blocks=6):
        super().__init__()
        layers = []
        C = in_channels
        for i in range(n_blocks):
            layers.append(ActNorm(C))
            layers.append(AffineCoupling(C))
            # simple channel permutation by reversing channels
            layers.append(ReverseChannels())
        self.layers = nn.ModuleList(layers)
        self.register_buffer('_prior_mean', torch.zeros(1))
        self.register_buffer('_prior_logstd', torch.zeros(1))

    def forward(self, x):
        # returns z and total logdet
        logdet_total = x.new_zeros(x.size(0))
        y = x
        for layer in self.layers:
            y, logdet = layer(y, reverse=False)
            logdet_total = logdet_total + logdet
        z = y
        return z, logdet_total

    def inverse(self, z):
        # invert layers in reverse order
        y = z
        for layer in reversed(self.layers):
            y, _ = layer(y, reverse=True)
        return y

    def log_prob(self, x):
        z, logdet = self.forward(x)
        # assume standard normal prior on flattened z
        # compute log p(z)
        z_flat = z.view(z.size(0), -1)
        logpz = -0.5 * (z_flat**2).sum(1) - 0.5 * z_flat.size(1) * torch.log(torch.tensor(2*torch.pi))
        return logpz + logdet

# small ReverseChannels layer
class ReverseChannels(nn.Module):
    def forward(self, x, reverse=False):
        if not reverse:
            y = x.flip(dims=[1])
            return y, x.new_zeros(x.size(0))
        else:
            y = x.flip(dims=[1])
            return y, x.new_zeros(x.size(0))

# ------------------------
# training sketch (MLE)
# ------------------------
def train_flow(flow, dataloader, epochs=10, lr=1e-3, device='cuda'):
    flow.to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    for ep in range(epochs):
        flow.train()
        total_nll = 0.0
        n = 0
        for xb, _ in dataloader:
            xb = xb.to(device)
            n += xb.size(0)
            logpx = flow.log_prob(xb)  # log p(x)
            loss = -logpx.mean()  # negative log likelihood
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_nll += loss.item() * xb.size(0)
        print(f"Epoch {ep+1}/{epochs} NLL: {total_nll / n:.4f}")

# ------------------------
# purification at test-time: optimize z
# ------------------------
def purify(flow, x_adv, g_val=0.5, steps=50, lr=1e-1, device='cuda'):
    """
    Given x_adv (tensor 1 x C x H x W), returns purified x_pure.
    g_val in [0,1] scalar (could be output of gating net).
    """
    flow.to(device)
    x_adv = x_adv.to(device)
    with torch.no_grad():
        z0, _ = flow.forward(x_adv)   # starting latent
    z = z0.detach().clone()
    z.requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    for i in range(steps):
        x_rec = flow.inverse(z)
        fidelity = F.mse_loss(x_rec, x_adv, reduction='sum')  # keep sum because we differentiate
        prior_reg = (z.view(z.size(0), -1).pow(2).sum(1)).mean()
        loss = fidelity + g_val * prior_reg
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        x_pure = flow.inverse(z)
    return x_pure.clamp(0,1)

# ------------------------
# example usage
# ------------------------
if __name__ == "__main__":
    # dummy dataset
    from torchvision import datasets, transforms
    ds = datasets.FakeData(size=1024, image_size=(3,32,32), transform=transforms.ToTensor())
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    flow = SimpleFlow(in_channels=3, n_blocks=4)
    train_flow(flow, dl, epochs=3, lr=1e-3, device='cpu')

    # take 1 sample, simulate attack by adding noise
    x, _ = ds[0]
    x_adv = (x + 0.2 * torch.randn_like(x)).clamp(0,1).unsqueeze(0)
    x_pure = purify(flow, x_adv, g_val=0.8, steps=200, lr=1e-2, device='cpu')
    print("done")
