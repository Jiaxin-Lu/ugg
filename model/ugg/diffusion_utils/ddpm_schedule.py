import numpy as np
import torch
import torch.nn as nn

from .diff_utils import stp


class DDPMSchedule(nn.Module):  # discrete time
    """This DDPM scheduler is designed for timestep 1..N
    """
    def __init__(self, _betas):
        r""" _betas[0...999] = betas[1...1000]
             for n>=1, betas[n] is the variance of q(xn|xn-1)
             for n=0,  betas[0]=0
        """
        super().__init__()
        assert _betas[0] < _betas[-1] < 1.

        betas = torch.cat([torch.zeros(1), _betas])
        alphas = 1. - betas
        self.N = len(_betas)

        assert isinstance(betas, torch.Tensor) and betas[0] == 0
        assert isinstance(alphas, torch.Tensor) and alphas[0] == 1
        assert len(betas) == len(alphas)

        skip_alphas, skip_betas = get_skip(alphas, betas)
        cum_alphas = skip_alphas[0]  # cum_alphas = alphas.cumprod()
        cum_betas = skip_betas[0]  # 1-cum_alphas
        
        sqrt_recip_alphas_cumprod = torch.sqrt(1. / cum_alphas)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / cum_alphas - 1.)

        posterior_variance = betas[1:] * (1. - cum_alphas[:-1]) / (1. - cum_alphas[1:])
        posterior_log_variance_clipped = torch.log(posterior_variance.clip(min=1e-20))
        posterior_mean_coef0 = betas[1:] * torch.sqrt(cum_alphas[:-1]) / (1. - cum_alphas[1:])
        posterior_mean_coeft = (1 - cum_alphas[:-1]) * torch.sqrt(alphas[1:]) / (1. - cum_alphas[1:])

        posterior_variance = torch.cat([torch.zeros(1), posterior_variance])
        posterior_log_variance_clipped = torch.cat([torch.zeros(1), posterior_log_variance_clipped])
        posterior_mean_coef0 = torch.cat([torch.zeros(1), posterior_mean_coef0])
        posterior_mean_coeft = torch.cat([torch.zeros(1), posterior_mean_coeft])

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('skip_alphas', skip_alphas)
        self.register_buffer('skip_betas', skip_betas)
        self.register_buffer('cum_alphas', cum_alphas)
        self.register_buffer('cum_betas', cum_betas)
        self.register_buffer('sqrt_recip_alphas_cumprod', sqrt_recip_alphas_cumprod)
        self.register_buffer('sqrt_recipm1_alphas_cumprod', sqrt_recipm1_alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', posterior_log_variance_clipped)
        self.register_buffer('posterior_mean_coef0', posterior_mean_coef0)
        self.register_buffer('posterior_mean_coeft', posterior_mean_coeft)
        
    def tilde_beta(self, s, t):
        return self.skip_betas[s, t] * self.cum_betas[s] / self.cum_betas[t]
    
    def sample(self, x0, t=None, eps=None):
        if t is None:
            t = np.random.choice(list(range(0, self.N + 1)), (len(x0),))
        if eps is None:
            eps = torch.randn_like(x0)
        eps[t == 0] = 0
        xn = stp(self.cum_alphas[t] ** 0.5, x0) + stp(self.cum_betas[t] ** 0.5, eps)
        t = torch.tensor(t, device=x0.device)
        return t, eps, xn
    
    # def sample_t_eps(self, x0, t, eps):
    #     xn = stp(self.cum_alphas[t] ** 0.5, x0) + stp(self.cum_betas[t] ** 0.5, eps)
    #     return xn
    
    # def sample_t(self, x0, t):
    #     eps = torch.randn_like(x0)
    #     xn = stp(self.cum_alphas[t] ** 0.5, x0) + stp(self.cum_betas[t] ** 0.5, eps)
    #     return eps, xn
    
    def calc_x0(self, eps, x_t, t):
        x0 = stp(self.sqrt_recip_alphas_cumprod[t], x_t) - stp(self.sqrt_recipm1_alphas_cumprod[t], eps)
        return x0.to(x_t.device)
    
    def calc_mean(self, x0, x_t, t):
        mean = stp(self.posterior_mean_coef0[t], x0) + stp(self.posterior_mean_coeft[t], x_t)
        return mean.to(x_t.device)
    
    def get_posterior_variance(self, t, x_ref):
        dim = x_ref.dim()
        extra_dims = (1,) * (dim - 1)
        post_v = self.posterior_variance[t].type_as(x_ref)
        return post_v.view(-1, *extra_dims)
    
    def get_posterior_log_variance(self, t, x_ref):
        dim = x_ref.dim()
        extra_dims = (1,) * (dim - 1)
        post_v = self.posterior_log_variance_clipped[t].type_as(x_ref)
        return post_v.view(-1, *extra_dims)
    
    def get_x_prev_and_pred_x0(self, eps, x_t, t):
        x0_pred = self.calc_x0(eps, x_t, t)
        mean_pred = self.calc_mean(x0_pred, x_t, t)
        # posterior_variance = self.get_posterior_variance(t, x_t)
        posterior_log_variance = self.get_posterior_log_variance(t, x_t)
        noise = torch.stack([torch.randn_like(x_t[i]) if t[i] > 1 else torch.zeros_like(x_t[i]) 
                             for i in range(x_t.shape[0])], dim=0)
        pred_x_ = mean_pred + (0.5 * posterior_log_variance).exp() * noise
        return pred_x_, x0_pred
    
    
def get_skip(alphas, betas):
    N = len(betas) - 1
    skip_alphas = torch.ones([N + 1, N + 1], dtype=betas.dtype)
    for s in range(N + 1):
        skip_alphas[s, s + 1:] = alphas[s + 1:].cumprod(dim=0)
    skip_betas = torch.zeros([N + 1, N + 1], dtype=betas.dtype)
    for t in range(N + 1):
        prod = betas[1: t + 1] * skip_alphas[1: t + 1, t]
        skip_betas[:t, t] = torch.flip(torch.flip(prod, dims=(0, )).cumsum(dim=0), dims=(0,))
    return skip_alphas, skip_betas


def diffusion_beta_schedule(start=0.00085, end=0.0120, n_timestep=1000, schedule='quad'):
    """
    diffusion beta scheduler, default: quad
    """
    if schedule == 'quad':
        _betas = (
            torch.linspace(start ** 0.5, end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )
    elif schedule == 'linear':
        _betas = torch.linspace(start, end, n_timestep, dtype=torch.float64)
    elif schedule == 'const':
        _betas = end * torch.ones(n_timestep, dtype=torch.float64)
    elif schedule == 'jsd':
        _betas = 1. / (torch.linspace(n_timestep, 1, n_timestep, dtype=torch.float64))
    else:
        raise NotImplementedError(f"schedule {schedule} not implemented")
    return _betas


