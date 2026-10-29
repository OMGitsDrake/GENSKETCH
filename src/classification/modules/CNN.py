import torch.nn as nn

class CNN_classification(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_layers: int,
        out_classes: int,
        starting_hidden_channels: int = 16
    ):
        super().__init__()
        self.hidden_layers = hidden_layers
        hcs = starting_hidden_channels
        
        self.blocks = nn.ModuleList()
        for i in range(hidden_layers):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=(in_channels if i == 0 else hcs // 2),
                        out_channels=hcs,
                        kernel_size=3,
                        stride=1, # (1 if i < (hidden_layers - 1) else 2),
                        padding=1
                    ),
                    nn.GroupNorm(
                        num_groups=4,
                        num_channels=hcs
                    ),
                    nn.LeakyReLU(),
                    nn.Conv2d(
                        in_channels=hcs,
                        out_channels=hcs,
                        kernel_size=3,
                        stride=1, # (1 if i < (hidden_layers - 1) else 2),
                        padding=1
                    ),
                    nn.GroupNorm(
                        num_groups=4,
                        num_channels=hcs
                    ),
                    nn.LeakyReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2) if i < (hidden_layers - 1) else nn.AdaptiveAvgPool2d(output_size=1)
                )
            )
            hcs *= 2

        self.flatten_layer = nn.Flatten()
        self.output_layer = nn.Linear(hcs//2, out_classes)

    def forward(self, minibatch):
        out = minibatch
        for block in self.blocks:
            out = block(out)

        flattened = self.flatten_layer(out)
        return self.output_layer(flattened)
