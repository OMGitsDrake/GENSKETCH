import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size: int = 65536, output_size: int = 125):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 300),
            nn.LeakyReLU(),
            nn.Linear(300, 300),
            nn.LeakyReLU(),
            nn.Linear(300, output_size),
        )

    def forward(self, x):
        return self.net(x)