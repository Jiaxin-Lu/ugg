# # Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# #
# # NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# # and proprietary rights in and to this software, related documentation
# # and any modifications thereto.  Any use, reproduction, disclosure or
# # distribution of this software and related documentation without an express
# # license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
# import torch
# import math
# import torch.nn as nn
# import numpy as np

# import os
# import math
# import shutil
# import json
# import time
# import sys
# import types
# from PIL import Image
# import torch
# import torch.nn as nn
# import numpy as np
# from torch import optim
# import torch.distributed as dist
# from torch.cuda.amp import autocast, GradScaler

# class PositionalEmbedding(nn.Module):
#     def __init__(self, embedding_dim, scale):
#         super(PositionalEmbedding, self).__init__()
#         self.embedding_dim = embedding_dim
#         self.scale = scale

#     def forward(self, timesteps):
#         assert len(timesteps.shape) == 1
#         timesteps = timesteps * self.scale
#         half_dim = self.embedding_dim // 2
#         emb = math.log(10000) / (half_dim - 1)
#         emb = torch.exp(torch.arange(half_dim) * -emb)
#         emb = emb.to(device=timesteps.device)
#         emb = timesteps[:, None] * emb[None, :]
#         emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
#         return emb


# class RandomFourierEmbedding(nn.Module):
#     def __init__(self, embedding_dim, scale):
#         super(RandomFourierEmbedding, self).__init__()
#         self.w = nn.Parameter(torch.randn(size=(1, embedding_dim // 2)) * scale, requires_grad=False)

#     def forward(self, timesteps):
#         emb = torch.mm(timesteps[:, None], self.w * 2 * 3.14159265359)
#         return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)


# def init_temb_fun(embedding_type, embedding_scale, embedding_dim):
#     if embedding_type == 'positional':
#         temb_fun = PositionalEmbedding(embedding_dim, embedding_scale)
#     elif embedding_type == 'fourier':
#         temb_fun = RandomFourierEmbedding(embedding_dim, embedding_scale)
#     else:
#         raise NotImplementedError

#     return temb_fun



# class DummyGradScalar(object):
#     def __init__(self, *args, **kwargs):
#         pass

#     def scale(self, input):
#         return input

#     def update(self):
#         pass

#     def state_dict(self):
#         return {}

#     def load_state_dict(self, x):
#         pass

#     def step(self, opt):
#         opt.step()

#     def unscale_(self, x):
#         return x


# class AvgrageMeter(object):

#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.avg = 0
#         self.sum = 0
#         self.cnt = 0

#     def update(self, val, n=1):
#         self.sum += val * n
#         self.cnt += n
#         self.avg = self.sum / self.cnt


# class ExpMovingAvgrageMeter(object):

#     def __init__(self, momentum=0.9):
#         self.momentum = momentum
#         self.reset()

#     def reset(self):
#         self.avg = 0

#     def update(self, val):
#         self.avg = (1. - self.momentum) * self.avg + self.momentum * val


# class DummyDDP(nn.Module):
#     def __init__(self, model):
#         super(DummyDDP, self).__init__()
#         self.module = model

#     def forward(self, *input, **kwargs):
#         return self.module(*input, **kwargs)

# def kl_per_group_vada(all_log_q, all_neg_log_p):
#     assert(len(all_log_q) == len(all_neg_log_p)
#            ), f'get len={len(all_log_q)} and {len(all_neg_log_p)}'

#     kl_all_list = []
#     kl_diag = []
#     for log_q, neg_log_p in zip(all_log_q, all_neg_log_p):
#         kl_diag.append(torch.mean(
#             torch.sum(neg_log_p + log_q, dim=[2, 3]), dim=0))
#         kl_all_list.append(torch.sum(neg_log_p + log_q,
#                            dim=[1, 2, 3]))  # sum over D,H,W

#     # kl_all = torch.stack(kl_all, dim=1)   # batch x num_total_groups
#     kl_vals = torch.mean(torch.stack(kl_all_list, dim=1),
#                          dim=0)   # mean per group

#     return kl_all_list, kl_vals, kl_diag


# def kl_coeff(step, total_step, constant_step, min_kl_coeff, max_kl_coeff):
#     # return max(min(max_kl_coeff * (step - constant_step) / total_step, max_kl_coeff), min_kl_coeff)
#     return max(min(min_kl_coeff + (max_kl_coeff - min_kl_coeff) * (step - constant_step) / total_step, max_kl_coeff), min_kl_coeff)


# def log_iw(decoder, x, log_q, log_p, crop=False):
#     recon = reconstruction_loss(decoder, x, crop)
#     return - recon - log_q + log_p


# def reconstruction_loss(decoder, x, crop=False):

#     recon = decoder.log_p(x)
#     if crop:
#         recon = recon[:, :, 2:30, 2:30]

#     return - torch.sum(recon, dim=[1, 2, 3])


# def sum_log_q(all_log_q):
#     log_q = 0.
#     for log_q_conv in all_log_q:
#         log_q += torch.sum(log_q_conv, dim=[1, 2, 3])

#     return log_q


# def tile_image(batch_image, n, m=None):
#     if m is None:
#         m = n
#     assert n * m == batch_image.size(0)
#     channels, height, width = batch_image.size(
#         1), batch_image.size(2), batch_image.size(3)
#     batch_image = batch_image.view(n, m, channels, height, width)
#     batch_image = batch_image.permute(2, 0, 3, 1, 4)  # n, height, n, width, c
#     batch_image = batch_image.contiguous().view(channels, n * height, m * width)
#     return batch_image


# def average_gradients_naive(params, is_distributed):
#     """ Gradient averaging. """
#     if is_distributed:
#         size = float(dist.get_world_size())
#         for param in params:
#             if param.requires_grad:
#                 param.grad.data /= size
#                 dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)


# def average_gradients(params, is_distributed):
#     """ Gradient averaging. """
#     if is_distributed:
#         if isinstance(params, types.GeneratorType):
#             params = [p for p in params]

#         size = float(dist.get_world_size())
#         grad_data = []
#         grad_size = []
#         grad_shapes = []
#         # Gather all grad values
#         for param in params:
#             if param.requires_grad:
#                 if param.grad is not None:
#                     grad_size.append(param.grad.data.numel())
#                     grad_shapes.append(list(param.grad.data.shape))
#                     grad_data.append(param.grad.data.flatten())
#         grad_data = torch.cat(grad_data).contiguous()

#         # All-reduce grad values
#         grad_data /= size
#         dist.all_reduce(grad_data, op=dist.ReduceOp.SUM)

#         # Put back the reduce grad values to parameters
#         base = 0
#         i = 0
#         for param in params:
#             if param.requires_grad and param.grad is not None:
#                 param.grad.data = grad_data[base:base +
#                                             grad_size[i]].view(grad_shapes[i])
#                 base += grad_size[i]
#                 i += 1


# def average_params(params, is_distributed):
#     """ parameter averaging. """
#     if is_distributed:
#         size = float(dist.get_world_size())
#         for param in params:
#             param.data /= size
#             dist.all_reduce(param.data, op=dist.ReduceOp.SUM)


# def average_tensor(t, is_distributed):
#     if is_distributed:
#         size = float(dist.get_world_size())
#         dist.all_reduce(t.data, op=dist.ReduceOp.SUM)
#         t.data /= size


# def broadcast_params(params, is_distributed):
#     if is_distributed:
#         for param in params:
#             dist.broadcast(param.data, src=0)


# def num_output(dataset):
#     if dataset in {'mnist',  'omniglot'}:
#         return 28 * 28
#     elif dataset == 'cifar10':
#         return 3 * 32 * 32
#     elif dataset.startswith('celeba') or dataset.startswith('imagenet') or dataset.startswith('lsun'):
#         size = int(dataset.split('_')[-1])
#         return 3 * size * size
#     elif dataset == 'ffhq':
#         return 3 * 256 * 256
#     else:
#         raise NotImplementedError


# def get_input_size(dataset):
#     if dataset in {'mnist', 'omniglot'}:
#         return 32
#     elif dataset == 'cifar10':
#         return 32
#     elif dataset.startswith('celeba') or dataset.startswith('imagenet') or dataset.startswith('lsun'):
#         size = int(dataset.split('_')[-1])
#         return size
#     elif dataset == 'ffhq':
#         return 256
#     elif dataset.startswith('shape'):
#         return 1  # 2048
#     else:
#         raise NotImplementedError


# def get_bpd_coeff(dataset):
#     n = num_output(dataset)
#     return 1. / np.log(2.) / n


# def get_channel_multiplier(dataset, num_scales):
#     if dataset in {'cifar10', 'omniglot'}:
#         mult = (1, 1, 1)
#     elif dataset in {'celeba_256', 'ffhq', 'lsun_church_256'}:
#         if num_scales == 3:
#             mult = (1, 1, 1)        # used for prior at 16
#         elif num_scales == 4:
#             mult = (1, 2, 2, 2)     # used for prior at 32
#         elif num_scales == 5:
#             mult = (1, 1, 2, 2, 2)  # used for prior at 64
#     elif dataset == 'mnist':
#         mult = (1, 1)
#     else:
#         mult = (1, 1)
#         # raise NotImplementedError

#     return mult


# def get_attention_scales(dataset):
#     if dataset in {'cifar10', 'omniglot'}:
#         attn = (True, False, False)
#     elif dataset in {'celeba_256', 'ffhq', 'lsun_church_256'}:
#         # attn = (False, True, False, False) # used for 32
#         attn = (False, False, True, False, False)  # used for 64
#     elif dataset == 'mnist':
#         attn = (True, False)
#     else:
#         raise NotImplementedError

#     return attn


# def change_bit_length(x, num_bits):
#     if num_bits != 8:
#         x = torch.floor(x * 255 / 2 ** (8 - num_bits))
#         x /= (2 ** num_bits - 1)
#     return x


# def view4D(t, size, inplace=True):
#     """
#      Equal to view(-1, 1, 1, 1).expand(size)
#      Designed because of this bug:
#      https://github.com/pytorch/pytorch/pull/48696
#     """
#     if inplace:
#         return t.unsqueeze_(-1).unsqueeze_(-1).unsqueeze_(-1).expand(size)
#     else:
#         return t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(size)


# def get_arch_cells(arch_type, use_se):
#     if arch_type == 'res_mbconv':
#         arch_cells = dict()
#         arch_cells['normal_enc'] = {'conv_branch': [
#             'res_bnswish', 'res_bnswish'], 'se': use_se}
#         arch_cells['down_enc'] = {'conv_branch': [
#             'res_bnswish', 'res_bnswish'], 'se': use_se}
#         arch_cells['normal_dec'] = {
#             'conv_branch': ['mconv_e6k5g0'], 'se': use_se}
#         arch_cells['up_dec'] = {'conv_branch': ['mconv_e6k5g0'], 'se': use_se}
#         arch_cells['normal_pre'] = {'conv_branch': [
#             'res_bnswish', 'res_bnswish'], 'se': use_se}
#         arch_cells['down_pre'] = {'conv_branch': [
#             'res_bnswish', 'res_bnswish'], 'se': use_se}
#         arch_cells['normal_post'] = {
#             'conv_branch': ['mconv_e3k5g0'], 'se': use_se}
#         arch_cells['up_post'] = {'conv_branch': ['mconv_e3k5g0'], 'se': use_se}
#         arch_cells['ar_nn'] = ['']
#     elif arch_type == 'res_bnswish':
#         arch_cells = dict()
#         arch_cells['normal_enc'] = {'conv_branch': [
#             'res_bnswish', 'res_bnswish'], 'se': use_se}
#         arch_cells['down_enc'] = {'conv_branch': [
#             'res_bnswish', 'res_bnswish'], 'se': use_se}
#         arch_cells['normal_dec'] = {'conv_branch': [
#             'res_bnswish', 'res_bnswish'], 'se': use_se}
#         arch_cells['up_dec'] = {'conv_branch': [
#             'res_bnswish', 'res_bnswish'], 'se': use_se}
#         arch_cells['normal_pre'] = {'conv_branch': [
#             'res_bnswish', 'res_bnswish'], 'se': use_se}
#         arch_cells['down_pre'] = {'conv_branch': [
#             'res_bnswish', 'res_bnswish'], 'se': use_se}
#         arch_cells['normal_post'] = {'conv_branch': [
#             'res_bnswish', 'res_bnswish'], 'se': use_se}
#         arch_cells['up_post'] = {'conv_branch': [
#             'res_bnswish', 'res_bnswish'], 'se': use_se}
#         arch_cells['ar_nn'] = ['']
#     elif arch_type == 'res_bnswish2':
#         arch_cells = dict()
#         arch_cells['normal_enc'] = {
#             'conv_branch': ['res_bnswish_x2'], 'se': use_se}
#         arch_cells['down_enc'] = {
#             'conv_branch': ['res_bnswish_x2'], 'se': use_se}
#         arch_cells['normal_dec'] = {
#             'conv_branch': ['res_bnswish_x2'], 'se': use_se}
#         arch_cells['up_dec'] = {'conv_branch': [
#             'res_bnswish_x2'], 'se': use_se}
#         arch_cells['normal_pre'] = {
#             'conv_branch': ['res_bnswish_x2'], 'se': use_se}
#         arch_cells['down_pre'] = {
#             'conv_branch': ['res_bnswish_x2'], 'se': use_se}
#         arch_cells['normal_post'] = {
#             'conv_branch': ['res_bnswish_x2'], 'se': use_se}
#         arch_cells['up_post'] = {'conv_branch': [
#             'res_bnswish_x2'], 'se': use_se}
#         arch_cells['ar_nn'] = ['']
#     elif arch_type == 'res_mbconv_attn':
#         arch_cells = dict()
#         arch_cells['normal_enc'] = {'conv_branch': [
#             'res_bnswish', 'res_bnswish', ], 'se': use_se, 'attn_type': 'attn'}
#         arch_cells['down_enc'] = {'conv_branch': [
#             'res_bnswish', 'res_bnswish'], 'se': use_se, 'attn_type': 'attn'}
#         arch_cells['normal_dec'] = {'conv_branch': [
#             'mconv_e6k5g0'], 'se': use_se, 'attn_type': 'attn'}
#         arch_cells['up_dec'] = {'conv_branch': [
#             'mconv_e6k5g0'], 'se': use_se, 'attn_type': 'attn'}
#         arch_cells['normal_pre'] = {'conv_branch': [
#             'res_bnswish', 'res_bnswish'], 'se': use_se}
#         arch_cells['down_pre'] = {'conv_branch': [
#             'res_bnswish', 'res_bnswish'], 'se': use_se}
#         arch_cells['normal_post'] = {
#             'conv_branch': ['mconv_e3k5g0'], 'se': use_se}
#         arch_cells['up_post'] = {'conv_branch': ['mconv_e3k5g0'], 'se': use_se}
#         arch_cells['ar_nn'] = ['']
#     elif arch_type == 'res_mbconv_attn_half':
#         arch_cells = dict()
#         arch_cells['normal_enc'] = {'conv_branch': [
#             'res_bnswish', 'res_bnswish'], 'se': use_se}
#         arch_cells['down_enc'] = {'conv_branch': [
#             'res_bnswish', 'res_bnswish'], 'se': use_se}
#         arch_cells['normal_dec'] = {'conv_branch': [
#             'mconv_e6k5g0'], 'se': use_se, 'attn_type': 'attn'}
#         arch_cells['up_dec'] = {'conv_branch': [
#             'mconv_e6k5g0'], 'se': use_se, 'attn_type': 'attn'}
#         arch_cells['normal_pre'] = {'conv_branch': [
#             'res_bnswish', 'res_bnswish'], 'se': use_se}
#         arch_cells['down_pre'] = {'conv_branch': [
#             'res_bnswish', 'res_bnswish'], 'se': use_se}
#         arch_cells['normal_post'] = {
#             'conv_branch': ['mconv_e3k5g0'], 'se': use_se}
#         arch_cells['up_post'] = {'conv_branch': ['mconv_e3k5g0'], 'se': use_se}
#         arch_cells['ar_nn'] = ['']
#     else:
#         raise NotImplementedError

#     return arch_cells


# def get_arch_cells_denoising(arch_type, use_se, apply_sqrt2):
#     if arch_type == 'res_mbconv':
#         arch_cells = dict()
#         arch_cells['normal_enc_diff'] = {
#             'conv_branch': ['mconv_e6k5g0_gn'], 'se': use_se}
#         arch_cells['down_enc_diff'] = {
#             'conv_branch': ['mconv_e6k5g0_gn'], 'se': use_se}
#         arch_cells['normal_dec_diff'] = {
#             'conv_branch': ['mconv_e6k5g0_gn'], 'se': use_se}
#         arch_cells['up_dec_diff'] = {
#             'conv_branch': ['mconv_e6k5g0_gn'], 'se': use_se}
#     elif arch_type == 'res_ho':
#         arch_cells = dict()
#         arch_cells['normal_enc_diff'] = {
#             'conv_branch': ['res_gnswish_x2'], 'se': use_se}
#         arch_cells['down_enc_diff'] = {
#             'conv_branch': ['res_gnswish_x2'], 'se': use_se}
#         arch_cells['normal_dec_diff'] = {
#             'conv_branch': ['res_gnswish_x2'], 'se': use_se}
#         arch_cells['up_dec_diff'] = {
#             'conv_branch': ['res_gnswish_x2'], 'se': use_se}
#     elif arch_type == 'res_ho_p1':
#         arch_cells = dict()
#         arch_cells['normal_enc_diff'] = {
#             'conv_branch': ['res_gnswish_x2_p1'], 'se': use_se}
#         arch_cells['down_enc_diff'] = {
#             'conv_branch': ['res_gnswish_x2_p1'], 'se': use_se}
#         arch_cells['normal_dec_diff'] = {
#             'conv_branch': ['res_gnswish_x2_p1'], 'se': use_se}
#         arch_cells['up_dec_diff'] = {
#             'conv_branch': ['res_gnswish_x2_p1'], 'se': use_se}
#     elif arch_type == 'res_ho_attn':
#         arch_cells = dict()
#         arch_cells['normal_enc_diff'] = {
#             'conv_branch': ['res_gnswish_x2'], 'se': use_se}
#         arch_cells['down_enc_diff'] = {
#             'conv_branch': ['res_gnswish_x2'], 'se': use_se}
#         arch_cells['normal_dec_diff'] = {
#             'conv_branch': ['res_gnswish_x2'], 'se': use_se}
#         arch_cells['up_dec_diff'] = {
#             'conv_branch': ['res_gnswish_x2'], 'se': use_se}
#     else:
#         raise NotImplementedError

#     for k in arch_cells:
#         arch_cells[k]['apply_sqrt2'] = apply_sqrt2

#     return arch_cells


# def groups_per_scale(num_scales, num_groups_per_scale):
#     g = []
#     n = num_groups_per_scale
#     for s in range(num_scales):
#         assert n >= 1
#         g.append(n)
#     return g


# def symmetrize_image_data(images):
#     return 2.0 * images - 1.0


# def unsymmetrize_image_data(images):
#     return (images + 1.) / 2.


# def normalize_symmetric(images):
#     """
#     Normalize images by dividing the largest intensity. Used for visualizing the intermediate steps.
#     """
#     b = images.shape[0]
#     m, _ = torch.max(torch.abs(images).view(b, -1), dim=1)
#     images /= (m.view(b, 1, 1, 1) + 1e-3)

#     return images


# @torch.jit.script
# def soft_clamp5(x: torch.Tensor):
#     # 5. * torch.tanh(x / 5.) <--> soft differentiable clamp between [-5, 5]
#     return x.div(5.).tanh_().mul(5.)


# @torch.jit.script
# def soft_clamp(x: torch.Tensor, a: torch.Tensor):
#     return x.div(a).tanh_().mul(a)


# class SoftClamp5(nn.Module):
#     def __init__(self):
#         super(SoftClamp5, self).__init__()

#     def forward(self, x):
#         return soft_clamp5(x)


# def override_architecture_fields(args, stored_args, logging):
#     # list of architecture parameters used in NVAE:
#     architecture_fields = ['arch_instance', 'num_nf', 'num_latent_scales', 'num_groups_per_scale',
#                            'num_latent_per_group', 'num_channels_enc', 'num_preprocess_blocks',
#                            'num_preprocess_cells', 'num_cell_per_cond_enc', 'num_channels_dec',
#                            'num_postprocess_blocks', 'num_postprocess_cells', 'num_cell_per_cond_dec',
#                            'decoder_dist', 'num_x_bits', 'log_sig_q_scale', 'latent_grad_cutoff',
#                            'progressive_output_vae', 'progressive_input_vae', 'channel_mult']

#     for f in architecture_fields:
#         if not hasattr(args, f) or getattr(args, f) != getattr(stored_args, f):
#             logging.info('Setting %s from loaded checkpoint', f)
#             setattr(args, f, getattr(stored_args, f))


# def init_processes(rank, size, fn, args, config):
#     """ Initialize the distributed environment. """
#     os.environ['MASTER_ADDR'] = args.master_address
#     os.environ['MASTER_PORT'] = '6020'
#     print('set MASTER_PORT: {}, MASTER_PORT: {}', os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])

#     print('init_process: rank={}, world_size={}', rank, size)
#     torch.cuda.set_device(args.local_rank)
#     dist.init_process_group(
#         backend='nccl', init_method='env://', rank=rank, world_size=size)
#     fn(args, config)
#     print('barrier: rank={}, world_size={}', rank, size)
#     dist.barrier()
#     print('skip destroy_process_group: rank={}, world_size={}', rank, size)
#     # dist.destroy_process_group()
#     print('skip destroy fini')


# def sample_rademacher_like(y):
#     return torch.randint(low=0, high=2, size=y.shape, device='cuda') * 2 - 1


# def sample_gaussian_like(y):
#     return torch.randn_like(y, device='cuda')


# def trace_df_dx_hutchinson(f, x, noise, no_autograd):
#     """
#     Hutchinson's trace estimator for Jacobian df/dx, O(1) call to autograd
#     """
#     if no_autograd:
#         # the following is compatible with checkpointing
#         torch.sum(f * noise).backward()
#         # torch.autograd.backward(tensors=[f], grad_tensors=[noise])
#         jvp = x.grad
#         trJ = torch.sum(jvp * noise, dim=[1, 2, 3])
#         x.grad = None
#     else:
#         jvp = torch.autograd.grad(f, x, noise, create_graph=False)[0]
#         trJ = torch.sum(jvp * noise, dim=[1, 2, 3])
#         # trJ = torch.einsum('bijk,bijk->b', jvp, noise)  # we could test if there's a speed difference in einsum vs sum

#     return trJ


# def calc_jacobian_regularization(pred_params, eps_t, dae, var_t, m_t, f_t, g2_t, var_N_t, args):
#     """
#     Calculates Jabobian regularization loss. For reference implementations, see
#     https://github.com/facebookresearch/jacobian_regularizer/blob/master/jacobian/jacobian.py or
#     https://github.com/cfinlay/ffjord-rnode/blob/master/lib/layers/odefunc.py.
#     """
#     # eps_t_jvp = eps_t.detach()
#     # eps_t_jvp = eps_t.detach().requires_grad_()
#     if args.no_autograd_jvp:
#         raise NotImplementedError(
#             "We have not implemented no_autograd_jvp for jacobian reg.")

#     jvp_ode_func_norms = []
#     alpha = torch.sigmoid(dae.mixing_logit.detach())
#     for _ in range(args.jac_reg_samples):
#         noise = sample_gaussian_like(eps_t)
#         jvp = torch.autograd.grad(
#             pred_params, eps_t, noise, create_graph=True)[0]

#         if args.sde_type in ['geometric_sde', 'vpsde', 'power_vpsde']:
#             jvp_ode_func = alpha * (noise * torch.sqrt(var_t) - jvp)
#             if not args.jac_kin_reg_drop_weights:
#                 jvp_ode_func = f_t / torch.sqrt(var_t) * jvp_ode_func
#         elif args.sde_type in ['sub_vpsde', 'sub_power_vpsde']:
#             sigma2_N_t = (1.0 - m_t ** 2) ** 2 + m_t ** 2
#             jvp_ode_func = noise * torch.sqrt(var_t) / (1.0 - m_t ** 4) - (
#                 (1.0 - alpha) * noise * torch.sqrt(var_t) / sigma2_N_t + alpha * jvp)
#             if not args.jac_kin_reg_drop_weights:
#                 jvp_ode_func = f_t * (1.0 - m_t ** 4) / \
#                     torch.sqrt(var_t) * jvp_ode_func
#         elif args.sde_type in ['vesde']:
#             jvp_ode_func = (1.0 - alpha) * noise * \
#                 torch.sqrt(var_t) / var_N_t + alpha * jvp
#             if not args.jac_kin_reg_drop_weights:
#                 jvp_ode_func = 0.5 * g2_t / torch.sqrt(var_t) * jvp_ode_func
#         else:
#             raise ValueError("Unrecognized SDE type: {}".format(args.sde_type))

#         jvp_ode_func_norms.append(jvp_ode_func.view(
#             eps_t.size(0), -1).pow(2).sum(dim=1, keepdim=True))

#     jac_reg_loss = torch.cat(jvp_ode_func_norms, dim=1).mean()
#     # jac_reg_loss = torch.mean(jvp_ode_func.view(eps_t.size(0), -1).pow(2).sum(dim=1))
#     return jac_reg_loss


# def calc_kinetic_regularization(pred_params, eps_t, dae, var_t, m_t, f_t, g2_t, var_N_t, args):
#     """
#     Calculates kinetic regularization loss. For a reference implementation, see
#     https://github.com/cfinlay/ffjord-rnode/blob/master/lib/layers/wrappers/cnf_regularization.py
#     """
#     # eps_t_kin = eps_t.detach()

#     alpha = torch.sigmoid(dae.mixing_logit.detach())
#     if args.sde_type in ['geometric_sde', 'vpsde', 'power_vpsde']:
#         ode_func = alpha * (eps_t * torch.sqrt(var_t) - pred_params)
#         if not args.jac_kin_reg_drop_weights:
#             ode_func = f_t / torch.sqrt(var_t) * ode_func
#     elif args.sde_type in ['sub_vpsde', 'sub_power_vpsde']:
#         sigma2_N_t = (1.0 - m_t ** 2) ** 2 + m_t ** 2
#         ode_func = eps_t * torch.sqrt(var_t) / (1.0 - m_t ** 4) - (
#             (1.0 - alpha) * eps_t * torch.sqrt(var_t) / sigma2_N_t + alpha * pred_params)
#         if not args.jac_kin_reg_drop_weights:
#             ode_func = f_t * (1.0 - m_t ** 4) / torch.sqrt(var_t) * ode_func
#     elif args.sde_type in ['vesde']:
#         ode_func = (1.0 - alpha) * eps_t * torch.sqrt(var_t) / \
#             var_N_t + alpha * pred_params
#         if not args.jac_kin_reg_drop_weights:
#             ode_func = 0.5 * g2_t / torch.sqrt(var_t) * ode_func
#     else:
#         raise ValueError("Unrecognized SDE type: {}".format(args.sde_type))

#     kin_reg_loss = torch.mean(ode_func.view(
#         eps_t.size(0), -1).pow(2).sum(dim=1))
#     return kin_reg_loss


# def different_p_q_objectives(iw_sample_p, iw_sample_q):
#     assert iw_sample_p in ['ll_uniform', 'drop_all_uniform', 'll_iw', 'drop_all_iw', 'drop_sigma2t_iw', 'rescale_iw',
#                            'drop_sigma2t_uniform']
#     assert iw_sample_q in ['reweight_p_samples', 'll_uniform', 'll_iw']
#     # Removed assert below. It may be stupid, but user can still do it. It may make sense for debugging purposes.
#     # assert iw_sample_p != iw_sample_q, 'It does not make sense to use the same objectives for p and q, but train ' \
#     #                                    'with separated q and p updates. To reuse the p objective for q, specify ' \
#     #                                    '"reweight_p_samples" instead (for the ll-based objectives, the ' \
#     #                                    'reweighting factor will simply be 1.0 then)!'
#     # In these cases, we reuse the likelihood-based p-objective (either the uniform sampling version or the importance
#     # sampling version) also for q.
#     if iw_sample_p in ['ll_uniform', 'll_iw'] and iw_sample_q == 'reweight_p_samples':
#         return False
#     # In these cases, we are using a non-likelihood-based objective for p, and hence definitly need to use another q
#     # objective.
#     else:
#         return True

# def mask_inactive_variables(x, is_active):
#     x = x * is_active
#     return x


# def common_x_operations(x, num_x_bits):
#     x = x[0] if len(x) > 1 else x
#     x = x.cuda()

#     # change bit length
#     x = change_bit_length(x, num_x_bits)
#     x = symmetrize_image_data(x)

#     return x


# def vae_regularization(args, vae_sn_calculator, loss_weight=None):
#     """
#         when using hvae_trainer, we pass args=None, and loss_weight value 
#     """
#     regularization_q, vae_norm_loss, vae_bn_loss, vae_wdn_coeff = 0., 0., 0., args.weight_decay_norm_vae if loss_weight is None else loss_weight
#     if loss_weight is not None or args.train_vae:
#         vae_norm_loss = vae_sn_calculator.spectral_norm_parallel()
#         vae_bn_loss = vae_sn_calculator.batchnorm_loss()
#         regularization_q = (vae_norm_loss + vae_bn_loss) * vae_wdn_coeff

#     return regularization_q, vae_norm_loss, vae_bn_loss, vae_wdn_coeff


# def dae_regularization(args, dae_sn_calculator, diffusion, dae, step, t, pred_params_p, eps_t_p, var_t_p, m_t_p, g2_t_p):
#     dae_wdn_coeff = args.weight_decay_norm_dae
#     dae_norm_loss = dae_sn_calculator.spectral_norm_parallel()
#     dae_bn_loss = dae_sn_calculator.batchnorm_loss()
#     regularization_p = (dae_norm_loss + dae_bn_loss) * dae_wdn_coeff

#     # Jacobian regularization
#     jac_reg_loss = 0.
#     if args.jac_reg_coeff > 0.0 and step % args.jac_reg_freq == 0:
#         f_t = diffusion.f(t).view(-1, 1, 1, 1)
#         var_N_t = diffusion.var_N(
#             t).view(-1, 1, 1, 1) if args.sde_type == 'vesde' else None
#         """
#         # Arash: Please remove the following if it looks correct to you, Karsten.
#         # jac_reg_loss = utils.calc_jacobian_regularization(pred_params, eps_t, dae, var_t, m_t, f_t, args)
#         if args.iw_sample_q in ['ll_uniform', 'll_iw']:
#             pred_params_jac_reg = torch.chunk(pred_params, chunks=2, dim=0)[0]
#             var_t_jac_reg, m_t_jac_reg, f_t_jac_reg = torch.chunk(var_t, chunks=2, dim=0)[0], \
#                                                       torch.chunk(m_t, chunks=2, dim=0)[0], \
#                                                       torch.chunk(f_t, chunks=2, dim=0)[0]
#             g2_t_jac_reg = torch.chunk(g2_t, chunks=2, dim=0)[0]
#             var_N_t_jac_reg = torch.chunk(var_N_t, chunks=2, dim=0)[0] if args.sde_type == 'vesde' else None
#         else:
#             pred_params_jac_reg = pred_params
#             var_t_jac_reg, m_t_jac_reg, f_t_jac_reg, g2_t_jac_reg, var_N_t_jac_reg = var_t, m_t, f_t, g2_t, var_N_t
#         jac_reg_loss = utils.calc_jacobian_regularization(pred_params_jac_reg, eps_t_p, dae, var_t_jac_reg, m_t_jac_reg,
#                                                           f_t_jac_reg, g2_t_jac_reg, var_N_t_jac_reg, args)
#         """
#         jac_reg_loss = calc_jacobian_regularization(pred_params_p, eps_t_p, dae, var_t_p, m_t_p,
#                                                     f_t, g2_t_p, var_N_t, args)
#         regularization_p += args.jac_reg_coeff * jac_reg_loss

#     # Kinetic regularization
#     kin_reg_loss = 0.
#     if args.kin_reg_coeff > 0.0:
#         f_t = diffusion.f(t).view(-1, 1, 1, 1)
#         var_N_t = diffusion.var_N(
#             t).view(-1, 1, 1, 1) if args.sde_type == 'vesde' else None
#         """
#         # Arash: Please remove the following if it looks correct to you, Karsten.
#         # kin_reg_loss = utils.calc_kinetic_regularization(pred_params, eps_t, dae, var_t, m_t, f_t, args)
#         if args.iw_sample_q in ['ll_uniform', 'll_iw']:
#             pred_params_kin_reg = torch.chunk(pred_params, chunks=2, dim=0)[0]
#             var_t_kin_reg, m_t_kin_reg, f_t_kin_reg = torch.chunk(var_t, chunks=2, dim=0)[0], \
#                                                       torch.chunk(m_t, chunks=2, dim=0)[0], \
#                                                       torch.chunk(f_t, chunks=2, dim=0)[0]
#             g2_t_kin_reg = torch.chunk(g2_t, chunks=2, dim=0)[0]
#             var_N_t_kin_reg = torch.chunk(var_N_t, chunks=2, dim=0)[0] if args.sde_type == 'vesde' else None
#         else:
#             pred_params_kin_reg = pred_params
#             var_t_kin_reg, m_t_kin_reg, f_t_kin_reg, g2_t_kin_reg, var_N_t_kin_reg = var_t, m_t, f_t, g2_t, var_N_t
#         kin_reg_loss = utils.calc_kinetic_regularization(pred_params_kin_reg, eps_t_p, dae, var_t_kin_reg, m_t_kin_reg,
#                                                          f_t_kin_reg, g2_t_kin_reg, var_N_t_kin_reg, args)
#         """
#         kin_reg_loss = calc_kinetic_regularization(pred_params_p, eps_t_p, dae, var_t_p, m_t_p,
#                                                    f_t, g2_t_p, var_N_t, args)
#         regularization_p += args.kin_reg_coeff * kin_reg_loss

#     return regularization_p, dae_norm_loss, dae_bn_loss, dae_wdn_coeff, jac_reg_loss, kin_reg_loss


# def update_vae_lr(args, global_step, warmup_iters, vae_optimizer):
#     if global_step < warmup_iters:
#         lr = args.trainer.opt.lr * float(global_step) / warmup_iters
#         for param_group in vae_optimizer.param_groups:
#             param_group['lr'] = lr
#         # use same lr if lr for local-dae is not specified


# def update_lr(args, global_step, warmup_iters, dae_optimizer, vae_optimizer, dae_local_optimizer=None):
#     if global_step < warmup_iters:
#         lr = args.learning_rate_dae * float(global_step) / warmup_iters
#         if args.learning_rate_mlogit > 0 and len(dae_optimizer.param_groups) > 1:
#             lr_mlogit = args.learning_rate_mlogit * \
#                 float(global_step) / warmup_iters
#             for i, param_group in enumerate(dae_optimizer.param_groups):
#                 if i == 0:
#                     param_group['lr'] = lr_mlogit
#                 else:
#                     param_group['lr'] = lr
#         else:
#             for param_group in dae_optimizer.param_groups:
#                 param_group['lr'] = lr
#         # use same lr if lr for local-dae is not specified
#         lr = lr if args.learning_rate_dae_local <= 0 else args.learning_rate_dae_local * \
#             float(global_step) / warmup_iters
#         if dae_local_optimizer is not None:
#             for param_group in dae_local_optimizer.param_groups:
#                 param_group['lr'] = lr

#         if args.train_vae:
#             lr = args.learning_rate_vae * float(global_step) / warmup_iters
#             for param_group in vae_optimizer.param_groups:
#                 param_group['lr'] = lr


# def start_meters():
#     tr_loss_meter = AvgrageMeter()
#     vae_recon_meter = AvgrageMeter()
#     vae_kl_meter = AvgrageMeter()
#     vae_nelbo_meter = AvgrageMeter()
#     kl_per_group_ema = AvgrageMeter()
#     return tr_loss_meter, vae_recon_meter, vae_kl_meter, vae_nelbo_meter, kl_per_group_ema


# def epoch_logging(args, writer, step, vae_recon_meter, vae_kl_meter, vae_nelbo_meter, tr_loss_meter, kl_per_group_ema):
#     average_tensor(vae_recon_meter.avg, args.distributed)
#     average_tensor(vae_kl_meter.avg, args.distributed)
#     average_tensor(vae_nelbo_meter.avg, args.distributed)
#     average_tensor(tr_loss_meter.avg, args.distributed)
#     average_tensor(kl_per_group_ema.avg, args.distributed)

#     writer.add_scalar('epoch/vae_recon', vae_recon_meter.avg, step)
#     writer.add_scalar('epoch/vae_kl', vae_kl_meter.avg, step)
#     writer.add_scalar('epoch/vae_nelbo', vae_nelbo_meter.avg, step)
#     writer.add_scalar('epoch/total_loss', tr_loss_meter.avg, step)
#     # add kl value per group to tensorboard
#     for i in range(len(kl_per_group_ema.avg)):
#         writer.add_scalar('kl_value/group_%d' %
#                           i, kl_per_group_ema.avg[i], step)

