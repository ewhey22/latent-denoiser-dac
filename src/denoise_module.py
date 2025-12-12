import torch
from torch import nn 


# Both "GRN" and "ConvNeXtBlock" are from https://github.com/YangAi520/APCodec/blob/main/models.py (July 2025).

class GRN(nn.Module):
    """Global Response Normalization for 1D sequences (applied channel-wise)."""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt-style residual block with depthwise conv + GRN."""
    def __init__(self,
                 dim: int,
                 intermediate_dim: int,
                 layer_scale_init_value: float = None):
        super().__init__()
        self.dwconv  = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm    = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act     = nn.GELU()
        self.grn     = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

        if layer_scale_init_value is not None:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        else:
            self.gamma = None

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)            # (B,C,T) → (B,T,C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.transpose(1, 2)            # back to (B,C,T)
        if self.gamma is not None:
            x = self.gamma.view(1, -1, 1) * x
        return residual + x


class Denoiser(nn.Module):
    """
    Latent-space ConvNeXt stack that predicts a residual f(x) and returns x + f(x).

    Args:
        latent_ch: Number of channels in the codec latent frames
            (DAC-16 kHz encoder output has 1024 channels).
        width: Width of a ConvNeXt block.
        n_blocks: Number of ConvNeXt blocks to stack.
    """

    def __init__(self, latent_ch: int = 1024, width: int = 768, n_blocks: int = 8):
        super().__init__()
        self.latent_ch = latent_ch
        self.stem = nn.Conv1d(latent_ch, width, kernel_size=1)
        blocks = [ConvNeXtBlock(width, intermediate_dim=width * 4)
                  for _ in range(n_blocks)]
        self.blocks = nn.Sequential(*blocks)
        self.final = nn.Conv1d(width, latent_ch, kernel_size=1)

        # ---- initialise to (almost) identity --------------------------------
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self, x):                 # x : (B, C_lat, T)
        y = self.stem(x)
        y = self.blocks(y)
        y = self.final(y)
        return x + y                      # global residual



class Discriminator(nn.Module):
    """
    Lightweight 1D discriminator operating on latent sequences. Uses per-frame
    channel mixing followed by dilated temporal depthwise convolutions.
    """
    def __init__(self, bins=1024, hidden=512, groups=16, n_tblocks=5):
        super().__init__()

        # --- per-frame channel mixing (depth 2) (pointwise)
        self.per_frame = nn.Sequential(
            nn.Conv1d(bins, hidden, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(hidden, hidden, 1, bias=False),
            nn.LeakyReLU(0.2, True),
        )

        # --- stack of residual temporal blocks (depthwise)
        tblocks = []
        dil = 1
        for _ in range(n_tblocks):
            tblocks.append(
                nn.Sequential(
                    nn.Conv1d(hidden, hidden, 5,
                              padding=2*dil,
                              dilation=dil,
                              groups=groups),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv1d(hidden, hidden, 1, bias=False)  # point-wise fuse
                )
            )
            dil *= 2                # exponential dilation 1,2,4…
        self.temporal = nn.ModuleList(tblocks)

        self.head = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(hidden, 1, 1)
        )

        # spectral norm everywhere
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.utils.spectral_norm(m)

    def forward(self, z, return_feats=False):
        feats = []
        x = self.per_frame(z)        # (B,hid,T)
        feats.append(x)

        for block in self.temporal:
            x = x + block(x)         # residual
            feats.append(x)

        x = self.head(x)             # (B,1,T)
        logit = x.mean(-1).squeeze(1)
        return (logit, feats) if return_feats else logit
