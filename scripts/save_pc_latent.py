import sys
import os
from os.path import join as pjoin
base_dir = os.path.dirname(__file__)
sys.path.append(pjoin(base_dir, '..'))
sys.path.append(pjoin(base_dir, '../LION/'))

import pickle
import yaml
from easydict import EasyDict as edict
import torch
import trimesh
import numpy as np


data_path = "../data/dexgraspnet"
mesh_path = "../data/meshdata"
pc_args = "../checkpoints/lion/aeb159h_hvae_lion_B32/cfg.yml"

scale_list = [0.06, 0.08, 0.1, 0.12, 0.15]

cfg = edict()
cfg.MODEL = edict()
cfg.MODEL.diffusion = edict()
cfg.MODEL.diffusion.pc_global_dim = 128
cfg.DATA = edict()
cfg.DATA.PC_NUM_POINTS = 2048


grasp_code_list = []
for code in os.listdir(data_path):
    if code.endswith('.npy'):
        grasp_code_list.append(code[:-4])
grasp_code_list.sort()
print(len(grasp_code_list))

device = torch.device('cuda:0')

import LION.vae_adain as vae_adain

with open(pc_args, 'r') as f:
    pc_args = edict(yaml.full_load(f))
pc_latent_model = vae_adain.Model(cfg, pc_args)

pc_checkpoint = "../checkpoints/lion/aeb159h_hvae_lion_B32/checkpoints/epoch_5999_iters_1667999.pt"

print('Load vae_checkpoint: {}', pc_checkpoint)
vae_ckpt = torch.load(pc_checkpoint)
vae_weight = vae_ckpt['model']
pc_latent_model.load_state_dict(vae_weight)

pc_latent_model = pc_latent_model.to(device=device)
pc_latent_model.eval()

for i, code in enumerate(grasp_code_list):
    print(code)
    mesh_grasp_dir = os.path.join(mesh_path, code, "coacd/decomposed.obj")

    mesh = trimesh.load(mesh_grasp_dir, force='mesh')
    points_ori_list = []
    for j in range(10):
        samples, fid = mesh.sample(2048, return_index=True)
        points_ori_list.append(samples)
        np.save(os.path.join(mesh_path, code, f"coacd/pc_2048_{j:03d}.npy"), samples)
    points_ori = np.stack(points_ori_list, axis=0)  # [10, N, 3]

    pc_latent_dict = {}
    points_tensor = []
    for sc in scale_list:
        points = torch.tensor(points_ori * sc, dtype=torch.float32)
        points_tensor.append(points)
    
    points_tensor = torch.cat(points_tensor, dim=0).to(device)  # [lx10, N, 3]
    
    points_tensor *= 6.6
    
    _, _, latent_list = pc_latent_model.encode(points_tensor)

    l = len(scale_list)
    latent_list_np = [[t.detach().cpu().numpy().reshape(l, 10, -1) for t in li] for li in latent_list]

    for j, sc in enumerate(scale_list):
        latent_sc = [[t[j] for t in li] for li in latent_list_np]
        pc_latent_dict[sc] = latent_sc
    
    for j in range(10):
        new_latent_dict = {}
        for k, v in pc_latent_dict.items():
            v_new = [[tt[j] for tt in t] for t in v]
            new_latent_dict[k] = v_new
        with open(os.path.join(mesh_path, code, f"coacd/pc_norm_latent_LION_{j:03d}.pk"), 'wb') as f:
            pickle.dump(new_latent_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    if i % 500 == 0:
        print(i)
    
