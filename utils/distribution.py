import torch

def sample_normal(mu, log_sigma=None, sigma=None, temperature=1.):
    if sigma is None:
        sigma = torch.exp(log_sigma)
    sigma = sigma * temperature
    rho = torch.randn_like(mu)
    z = rho * sigma + mu
    return z, rho
