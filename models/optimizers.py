from functools import partial
import torch


optimizers = {
    'Adam': partial(torch.optim.Adam, betas=(0.5, 0.9)),
    'SGD': torch.optim.SGD,
}


def get_optimizer(optim_type: str):
    return optimizers[optim_type]
