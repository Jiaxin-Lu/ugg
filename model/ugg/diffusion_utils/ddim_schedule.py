import numpy as np
import torch
import torch.nn as nn

from .ddpm_schedule import DDPMSchedule
from .diff_utils import stp


class DDIMSchedule(nn.Module):
    def __init__(self, ddpm_schedule: DDPMSchedule, ddim_n_steps: int, ddim_discretize: str='uniform', ddim_eta: float=0.):
        super().__init__()
        self.ddpm_n_steps = ddpm_schedule.N
        self.ddim_n_steps = ddim_n_steps
        if ddim_discretize == 'uniform':
            c = self.ddpm_n_steps // self.ddim_n_steps
            self.time_steps = np.asarray(list(range(0, self.ddpm_n_steps, c))) + 2
        elif ddim_discretize == 'quad':
            self.time_steps = ((np.linspace(0, np.sqrt(self.ddpm_n_steps * .8), self.ddim_n_steps)) ** 2).astype(np.int64) + 2
        else:
            raise NotImplementedError(ddim_discretize)
        
        alpha_bar = ddpm_schedule.cum_alphas
        ddim_alpha = alpha_bar[self.time_steps].clone().to(torch.float32)
        ddim_alpha_sqrt = torch.sqrt(ddim_alpha)
        ddim_alpha_prev = torch.cat([alpha_bar[1:2], alpha_bar[self.time_steps[:-1]]])
        ddim_sigma = (ddim_eta * ((1 - ddim_alpha_prev) / (1 - ddim_alpha) *
                                       (1 - ddim_alpha / ddim_alpha_prev)) ** .5)
        ddim_sqrt_one_minus_alpha = (1. - ddim_alpha) ** .5

        self.register_buffer('ddim_alpha', ddim_alpha)
        self.register_buffer('ddim_alpha_sqrt', ddim_alpha_sqrt)
        self.register_buffer('ddim_alpha_prev', ddim_alpha_prev)
        self.register_buffer('ddim_sigma', ddim_sigma)
        self.register_buffer('ddim_sqrt_one_minus_alpha', ddim_sqrt_one_minus_alpha)

    def get_x_prev_and_pred_x0(self, eps, x_t, index, temperature: float = 1.):
        if index < 0:
            return x_t, None
        alpha = self.ddim_alpha[index]
        alpha_prev = self.ddim_alpha_prev[index]
        sigma = self.ddim_sigma[index]
        sqrt_one_minus_alpha = self.ddim_sqrt_one_minus_alpha[index]
        pred_x0 = (x_t - stp(sqrt_one_minus_alpha, eps)) / (alpha ** 0.5)
        dir_xt = (1. - alpha_prev - sigma ** 2).sqrt() * eps
        if sigma == 0:
            noise = 0.
        else:
            noise = torch.randn(x_t.shape, device=x_t.device)
        noise = noise * temperature
        x_prev = (alpha_prev ** 0.5) * pred_x0 + dir_xt + sigma * noise
        return x_prev, pred_x0
