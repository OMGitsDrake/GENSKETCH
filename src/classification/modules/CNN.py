import torch.nn as nn
from torch import cat
from .Residual import ResBlock
from ...utils import config as cfg

class CNN_classification(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_layers: int,
        out_classes: int,
        starting_hidden_channels: int = cfg.STARTING_HIDDEN_CHANNELS
    ):
        super().__init__()
        self.hidden_layers = hidden_layers
        hcs = starting_hidden_channels
        
        self.blocks = nn.ModuleList()
        for i in range(hidden_layers):
            input_channels = in_channels if i == 0 else hcs // 2
            
            self.blocks.append(
                nn.Sequential(                    
                    ResBlock(input_channels, hcs),
                    
                    nn.MaxPool2d(kernel_size=2, stride=2) if i < (hidden_layers - 1) else nn.Identity(),
                )
            )
            hcs *= 2

        self.dropout_2d = nn.Dropout2d(p=cfg.DROPOUT2D)

        self.max_block = nn.AdaptiveMaxPool2d(output_size=1)
        self.avg_block = nn.AdaptiveAvgPool2d(output_size=1)

        self.output_block = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=cfg.DROPOUT),
            nn.Linear(hcs, out_classes)
        )

    def forward(self, minibatch):
        out = minibatch
        for block in self.blocks:
            out = block(out)

        out = self.dropout_2d(out)

        out = cat([self.max_block(out), self.avg_block(out)], dim=1)

        out = self.output_block(out)
        return out
