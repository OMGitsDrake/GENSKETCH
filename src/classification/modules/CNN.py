import torch.nn as nn
import torch

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
        
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.non_linear_layers = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()
        for i in range(hidden_layers):
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=(in_channels if i == 0 else hcs // 2),
                    out_channels=hcs,
                    kernel_size=3,
                    stride=1, # (1 if i < (hidden_layers - 1) else 2),
                    padding=1
                )
            )

            self.norm_layers.append(
                # nn.BatchNorm2d(
                #     num_features=hcs
                # )
                nn.GroupNorm(
                    num_groups=4,
                    num_channels=hcs
                )
            )
            hcs *= 2

            self.non_linear_layers.append(
                nn.LeakyReLU()
            )

            if i < (hidden_layers - 1):
                self.pooling_layers.append(
                    nn.MaxPool2d(
                        kernel_size=2,
                        stride=2
                    )
                )
            else:
                self.pooling_layers.append(
                    nn.AdaptiveAvgPool2d(
                        output_size=1
                    )
                )

        self.flatten_layer = nn.Flatten()
        self.output_layer = nn.Linear(hcs//2, out_classes)

    def forward(self, minibatch):
        for i in range(self.hidden_layers):
            if i == 0:
                out = self.conv_layers[i](minibatch)
            else:
                out = self.conv_layers[i](out)

            out = self.norm_layers[i](out)
            out = self.non_linear_layers[i](out)
            out = self.pooling_layers[i](out)

        flattened = self.flatten_layer(out)
        return self.output_layer(flattened)
