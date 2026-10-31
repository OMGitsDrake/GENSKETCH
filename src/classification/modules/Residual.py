import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            )
    
            self.gn1 = nn.GroupNorm(
                num_groups=4,
                num_channels=out_channels
            )
    
            self.conv2 = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            )
    
            self.gn2 = nn.GroupNorm(
                num_groups=4,
                num_channels=out_channels
            )
    
            if in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                    ),
                    nn.GroupNorm(
                        num_groups=4,
                        num_channels=out_channels
                    )
                )
            else:
                self.shortcut = nn.Identity()

            self.activation = nn.LeakyReLU()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.gn2(out)

        out += identity
        out = self.activation(out)

        return out