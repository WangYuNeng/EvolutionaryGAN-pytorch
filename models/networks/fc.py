import torch
from torch import nn


class FCGenerator(nn.Module):

    def __init__(self, dim=300):
        super().__init__()
        self.layer = nn.Linear(dim, dim, bias=False)

    def forward(self, x: dict):
        x = x['source']
        return self.layer(x)


class FCDiscriminator(nn.Module):

    def __init__(self, dim=300, n_layers=2, n_hidden=2048):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(dim, n_hidden),
            *[
                nn.Linear(n_hidden, n_hidden)
                for _ in range(n_layers - 2)
            ],
        ])
        self.out = nn.Linear(n_hidden, 1)

    def forward(self, x: dict):
        x = x['data']
        for layer in self.layers:
            x = layer(x)
            x = torch.relu(x)
        logits = self.out(x)
        return logits
