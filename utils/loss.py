import torch


def kl_divergence(mu, logvar, reduction='mean'):
    if reduction == 'mean':
        return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
    elif reduction == 'mean_mean':
        return torch.mean(-0.5 * (1 + logvar - mu ** 2 - logvar.exp()))
    else:
        return -0.5 * (1 + logvar - mu ** 2 - logvar.exp())
    