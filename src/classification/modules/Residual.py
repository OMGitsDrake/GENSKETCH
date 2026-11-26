import torch.nn as nn
from ...utils import config as cfg

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
            super().__init__()
            if cfg.CONVS_PER_BLOCK <= 0 or not cfg.RESIDUAL_BLOCKS:
                raise ValueError("ResBlock richiede almeno una convoluzione e connessione residua")
            if cfg.NORMALIZATION != "GroupNorm":
                raise ValueError("ResBlock supporta soltanto GroupNorm")
            
            self.convolutions = nn.ModuleList()
            for _ in range(cfg.CONVS_PER_BLOCK):
                 self.convolutions.append(
                    nn.Conv2d(
                        in_channels=in_channels if _ == 0 else out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=1,
                    )
                 )
            self.gns = nn.ModuleList()
            for _ in range(cfg.CONVS_PER_BLOCK):
                 self.gns.append(
                      nn.GroupNorm(
                        num_groups=cfg.GROUPS,
                        num_channels=out_channels
                      )
                 )

            if in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                    ),
                    nn.GroupNorm(
                        num_groups=cfg.GROUPS,
                        num_channels=out_channels
                    )
                )
            else:
                self.shortcut = nn.Identity()

            self.activation = nn.LeakyReLU()

    def forward(self, x):
        out = x
        identity = self.shortcut(out)

        for i, (c, g) in enumerate(zip(self.convolutions, self.gns)):
            out = g(c(out))
            if i < len(self.convolutions) - 1:
                 out = self.activation(out)

        out = out +identity
        out = self.activation(out)

        return out
