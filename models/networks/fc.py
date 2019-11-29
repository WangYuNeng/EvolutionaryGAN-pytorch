import torch
from torch import nn

from .expm import expm


class FCGenerator(nn.Module):

    def __init__(self, dim=300):
        super().__init__()
        self.layer = nn.Parameter(torch.zeros(size=(dim, dim)), requires_grad=True)

    def forward(self, x: dict):
        x = x['source']
        triu = self.layer.triu()
        skew_symmetric_matrix = triu - triu.t()
        mapping = expm(skew_symmetric_matrix)
        return x @ mapping


class FCDiscriminator(nn.Module):

    def __init__(self, dim=300, n_layers=2, n_hidden=2048):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(dim, n_hidden),
            *[
                nn.Linear(n_hidden, n_hidden)
                for _ in range(n_layers - 1)
            ],
        ])
        self.out = nn.Linear(n_hidden, 1)

    def forward(self, x: dict):
        x = x['data']
        for layer in self.layers:
            x = nn.functional.dropout(x, 0.1)
            x = layer(x)
            x = nn.functional.leaky_relu(x, 0.2)
        logits = self.out(x)
        return logits
