import torch


def get_prior(bs, z_dim, z_type, device):
    if z_type == 'Gaussian':
        z = torch.randn(bs, z_dim, 1, 1, device=device)
    elif z_type == 'Uniform':
        z = torch.rand(bs, z_dim, 1, 1, device=device) * 2. - 1.
    return z
