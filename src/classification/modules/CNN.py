import torch.nn as nn
from .Residual import ResBlock

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
            input_channels = in_channels if i == 0 else hcs // 2
            
            self.blocks.append(
                nn.Sequential(                    
                    ResBlock(input_channels, hcs),
                    
                    nn.MaxPool2d(kernel_size=2, stride=2) if i < (hidden_layers - 1) else nn.AdaptiveAvgPool2d(output_size=1),
                )
            )
            hcs *= 2

        self.output_block = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(hcs//2, out_classes)
        )

    def forward(self, minibatch):
        out = minibatch
        for block in self.blocks:
            out = block(out)

        out = self.output_block(out)
        return out
