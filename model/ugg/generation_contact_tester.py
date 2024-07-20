import os
import sys
from os.path import join as pjoin

from model import build_model

base_dir = os.path.dirname(__file__)
sys.path.append(pjoin(base_dir, '..'))
sys.path.append(pjoin(base_dir, '..', '..'))

import pickle

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import yaml
from easydict import EasyDict as edict
from loguru import logger

from LION import vae_adain
from model.ugg.UViT.pn2_layer import square_distance
from utils.distribution import sample_normal
from utils.hand_helper import ROT_DIM_DICT, compose_hand_param
from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.utils import dict_to_device, extract_batch_num

torch.set_float32_matmul_precision('high')

LATENT_NAME_LIST = ['pc_global', 'pc_local', 'hand_param', 'hand_R', 'hand_t', 'contact_map']

class UGGGenerationTester(nn.Module):
    def __init__(self, cfg, ckpt=None, device='cpu'):
        super(UGGGenerationTester, self).__init__()
        self.cfg = cfg
        self.model_cfg = cfg.MODEL
        self.diffusion_cfg = cfg.MODEL.diffusion
        
        self.apply_random_rot = cfg.DATA.APPLY_RANDOM_ROT
        print("apply random rotation", self.apply_random_rot)
        
        self.original_diffusion = build_model(cfg=cfg, ckpt=ckpt)
        logger.info(f"load model from ckpt: {ckpt}")

        self.pc_latent_model = self.original_diffusion.pc_latent_model

        for name, para in self.pc_latent_model.named_parameters():
            para.requires_grad = False

        self.hand_latent_model = self.original_diffusion.hand_latent_model
        if self.hand_latent_model is None:
            logger.info("No hand encode")
        self.unet = self.original_diffusion.unet
        self.schedule = self.original_diffusion.schedule
        self.n_timestep = self.diffusion_cfg.n_timestep

        self.pc_latent_model.eval()
        if self.hand_latent_model is not None:
            self.hand_latent_model.eval()
        self.unet.eval()

        self.original_diffusion = None

        self.pc_global_dim = self.diffusion_cfg.pc_global_dim
        self.hand_param_dim = self.diffusion_cfg.hand_param_dim
        self.pc_local_pts = self.diffusion_cfg.pc_local_pts
        self.pc_local_pts_dim = self.diffusion_cfg.pc_local_pts_dim
        self.contact_num = 5

        self.contact_map_normalize_factor = self.diffusion_cfg.contact_map_normalize_factor
        self.normalize_pc = self.cfg.DATA.NORMALIZE_PC

        self._set_hand_rot()
        self._prep_generation()
        self._set_gen()

        self.hand_model = HandModel(
                mjcf_path='hand_model_mjcf/shadow_hand_wrist_free.xml',
                mesh_path='hand_model_mjcf/meshes',
                contact_points_path='hand_model_mjcf/contact_points.json',
                penetration_points_path='hand_model_mjcf/penetration_points.json',
                device=device
            )
        
        self.device = device

        self.task = self.diffusion_cfg.task
    
    def _set_gen(self):
        self.gen_hand = self.diffusion_cfg.gen_hand
        self.gen_pc = self.diffusion_cfg.gen_pc
        self.gen_contact = self.diffusion_cfg.gen_contact
        
    def get_generation_dict(self, B=100):
        pc_global_latent_gen = torch.randn(B, self.pc_global_dim)
        pc_local_latent_gen = torch.randn(B, self.pc_local_pts * self.pc_local_pts_dim)
        contact_map_gen = torch.randn(B, self.contact_num * 3)
        hand_param_latent_gen = torch.randn(B, self.hand_param_dim)
        hand_R_gen = torch.randn(B, self.hand_rot_dim)
        hand_t_gen = torch.randn(B, 3)
        gen_dict = {
            'hand_param_latent_gen': hand_param_latent_gen,
            'hand_R_gen': hand_R_gen,
            'hand_t_gen': hand_t_gen,
            'pc_global_latent_gen': pc_global_latent_gen,
            'pc_local_latent_gen': pc_local_latent_gen,
            'contact_map_gen': contact_map_gen,
        }
        return gen_dict
        
    def _prep_generation(self):
        self.gen_dict = self.get_generation_dict(100)

    def _set_hand_rot(self):
        self.hand_rot_type = self.cfg.DATA.ROT_TYPE.lower()
        
        self.hand_rot_dim = ROT_DIM_DICT[self.hand_rot_type]

        self.hand_trans_slice = slice(0, 3)
        self.hand_rot_slice = slice(3, 3+self.hand_rot_dim)
        self.hand_param_slice = slice(3+self.hand_rot_dim, None)

    @torch.no_grad()
    def on_after_batch_transfer(self, batch, dataloader_idx):
        # if apply random rot during training, need to apply the same random rot to the object pc
        if self.apply_random_rot and 'object_pc' in batch:
            r_rand = batch['r_rand']  # B, 3, 3
            object_pc = batch['object_pc'] # B, 3
            object_pc = torch.bmm(object_pc, r_rand)
            batch['object_pc'] = object_pc
            
        # compute point cloud latent
        if 'z_mu_global' in batch:
            z_mu_global = batch['z_mu_global']
            z_sigma_global = batch['z_sigma_global']
            z_mu_local = batch['z_mu_local']
            z_sigma_local = batch['z_sigma_local']
            pc_global_latent = sample_normal(mu=z_mu_global, log_sigma=z_sigma_global)[0]
            pc_local_latent = sample_normal(mu=z_mu_local, log_sigma=z_sigma_local)[0]
            batch['pc_global_latent'] = pc_global_latent
            batch['pc_local_latent'] = pc_local_latent
        elif 'object_pc' in batch:
            obj_pc = batch['object_pc']
            pc_vae_output = self.compute_pc_vae(obj_pc)
            pc_global_latent = pc_vae_output['eps_global']  # [B, D1]
            pc_local_latent = pc_vae_output['eps_local']  # [B, ND2]
            batch['pc_global_latent'] = pc_global_latent
            batch['pc_local_latent'] = pc_local_latent
            
        # compute hand parameter latent
        if 'hand_pose' in batch:
            hand_pose = batch['hand_pose']
            hand_param = hand_pose[:, self.hand_param_slice]
            hand_vae_output = self.compute_hand_vae(hand_param)
            hand_param_latent = hand_vae_output['eps']
            batch['hand_param_latent'] = hand_param_latent
        
        # compute contact map
        if self.gen_contact and 'ori_hand_pose' in batch and 'object_pc' in batch:
            contact_map, ori_contact_map = self.compute_contact_map(batch)
            batch['contact_map'] = contact_map
        return batch
    
    def normalize_pcd(self, pcd):
        if self.normalize_pc:
            return pcd * 6.6
        else:
            return pcd
        
    def normalize_back_pcd(self, pcd):
        if self.normalize_pc:
            return pcd / 6.6
        else:
            return pcd
        
    @torch.no_grad()
    def compute_contact_map(self, data_dict):
        if self.hand_model is None:
            self.hand_model = HandModel(
                mjcf_path='hand_model_mjcf/shadow_hand_wrist_free.xml',
                mesh_path='hand_model_mjcf/meshes',
                contact_points_path='hand_model_mjcf/contact_points.json',
                penetration_points_path='hand_model_mjcf/penetration_points.json',
                n_surface_points=1024,
                device=self.device,
            )
        self.hand_model.set_parameters(data_dict['ori_hand_pose'])
        hand_pts = self.hand_model.get_surface_points()
        pc = data_dict['object_pc']
        pc = self.normalize_back_pcd(pc)
        # the input pc is already normalized. 
        # we need original pc for contact map.
        dist = self.hand_model.cal_distance(pc)  # [B, N]
        B = dist.shape[0]
        contact_list = []
        for b in range(B):
            rg = torch.max(dist[b])
            rg = min(-0.0002, rg - 0.0001)
            pos = torch.where(dist[b] > rg)[0]
            pos = pos[torch.randint(pos.shape[0], (self.contact_num,))]
            contact_b = pc[b, pos]
            contact_list.append(contact_b)  # [5, 3]
        
        contact_map = torch.stack(contact_list, dim=0)
        contact_map = self.normalize_pcd(contact_map)

        ori_contact_map = 3. - 4. * torch.sigmoid(self.contact_map_normalize_factor * dist.abs())
        ori_contact_map = ori_contact_map.detach()
        return contact_map, (ori_contact_map, hand_pts)
    
    @torch.no_grad()
    def compute_pc_vae(self, pcd, **kwargs):
        all_eps, _, _ = self.pc_latent_model.encode(pcd)
        eps_global, eps_local = all_eps
        out_dict = {
            'eps_global': eps_global,
            'eps_local': eps_local,
        }
        return out_dict
    
    @torch.no_grad()
    def compute_hand_vae(self, hand_param, **kwargs):
        if self.hand_latent_model is None:
            out_dict = {
                'eps': hand_param
            }
        else:
            hand_out_dict = self.hand_latent_model.encode(dict(x=hand_param))
            eps = hand_out_dict['z']
            out_dict = {
                'eps': eps,
            }
        return out_dict
    
    @torch.no_grad()
    def decode_hand_vae(self, hand_param_latent, **kwargs):
        if self.hand_latent_model is None:
            return hand_param_latent
        else:
            hand_param_final = self.hand_latent_model.decode(dict(z=hand_param_latent))['x_recon']
            return hand_param_final
        
    @torch.no_grad()
    def decode_pc_vae(self, pc_global_latent, pc_local_latent, **kwargs):
        B = pc_global_latent.shape[0]
        pc_final = self.pc_latent_model.sample(num_samples=B, decomposed_eps=[pc_global_latent, pc_local_latent])
        return pc_final
    
    def forward(self, data_dict):
        pass
    
    @torch.no_grad()
    def p_sample(self, x_t, t_pc, t_hand, t_contact=None, *args, **kwargs):
        unet_dict = {
            k: x_t[k] for k in LATENT_NAME_LIST
        }
        unet_dict['t_pc'] = t_pc
        unet_dict['t_hand'] = t_hand
        unet_dict['t_contact'] = t_contact
        pred_eps = self.unet(unet_dict)
        t_pc_np = t_pc.detach().cpu().numpy()
        t_hand_np = t_hand.detach().cpu().numpy()
        pred_x = {}
        for k in LATENT_NAME_LIST:
            if 'contact' in k and not self.gen_contact:
                pred_x[k] = None
                continue
            if 'pc' in k:
                t = t_pc_np
            elif 'contact' in k:
                t = t_contact.detach().cpu().numpy()
            else:
                t = t_hand_np
            eps = pred_eps[f'{k}_eps_pred']
            x_t_in = x_t[k]
            pred_x_, x0_pred = self.schedule.get_x_prev_and_pred_x0(eps, x_t_in, t)
            pred_x[k] = pred_x_

        return pred_x
    
    
    @torch.no_grad()
    def p_sample_joint(self, data_dict):
        assert self.gen_hand and self.gen_pc, 'both gen_hand and gen_pc should be true'

        hand_param_latent_gen = data_dict['hand_param_latent_gen'].clone()
        hand_R_gen = data_dict['hand_R_gen'].clone()
        hand_t_gen = data_dict['hand_t_gen'].clone()
        pc_global_latent_gen = data_dict['pc_global_latent_gen'].clone()
        pc_local_latent_gen = data_dict['pc_local_latent_gen'].clone()
        B = hand_param_latent_gen.shape[0]

        x_t_dict = {
            'hand_param': hand_param_latent_gen,
            'hand_R': hand_R_gen,
            'hand_t': hand_t_gen,
            'pc_global': pc_global_latent_gen,
            'pc_local': pc_local_latent_gen,
        }
        x_t_dict['contact_map'] = None
        if self.gen_contact:
            x_t_dict['contact_map'] = data_dict['contact_map_gen']

        all_x_t = [x_t_dict]

        for t in reversed(range(1, self.n_timestep+1)):
            t_ = torch.ones(B, dtype=torch.long, device=self.device) * t
            if self.gen_contact:
                t_contact = t_
            else:
                t_contact = None
            x_t_new = self.p_sample(x_t_dict, t_hand=t_, t_pc=t_,t_contact=t_contact)
            all_x_t.append(x_t_new)
            x_t_dict = x_t_new
            
        final_dict = all_x_t[-1]
        hand_param_latent = final_dict['hand_param'].clone()

        hand_param_final = self.decode_hand_vae(hand_param_latent)
        
        pc_global_latent = final_dict['pc_global']
        pc_local_latent = final_dict['pc_local']
        
        pc_final = self.decode_pc_vae(pc_global_latent, pc_local_latent)

        pc_gen_list = []
        hand_param_gen_list = []
        hand_t_gen_list = []
        hand_R_gen_list = []
        for d in all_x_t:
            t_pc_global, t_pc_local = d['pc_global'], d['pc_local']
            pc_gen_list.append(self.decode_pc_vae(t_pc_global, t_pc_local))
            hand_R_gen_list.append(d['hand_R'])
            hand_t_gen_list.append(d['hand_t'])
            hand_param_gen_list.append(self.decode_hand_vae(d['hand_param']))
        
        ret_dict = {
            'pc_final': pc_final,
            'hand_param_final': hand_param_final,
            'hand_param_undecoded': final_dict['hand_param'],
            'hand_R_final': final_dict['hand_R'],
            'hand_t_final': final_dict['hand_t'],
            'hand_param_gen_list': hand_param_gen_list,
            'hand_t_gen_list': hand_t_gen_list,
            "hand_R_gen_list": hand_R_gen_list,
            'pc_gen_list': pc_gen_list,
        }

        if self.gen_contact:
            contact_map_gen_list = []
            for d in all_x_t:
                contact_map_gen_list.append(d['contact_map'])
            ret_dict["contact_map_final"] = final_dict['contact_map'].reshape(B, self.contact_num, 3)
            ret_dict["contact_map_gen_list"] = contact_map_gen_list
            
            hand_pose = compose_hand_param(final_dict['hand_t'], final_dict['hand_R'], hand_param_final, rot_type=self.hand_rot_type)
            contact_map_gt, contact_map_ori = self.compute_contact_map(dict(object_pc=pc_final, ori_hand_pose=hand_pose))
            ret_dict['contact_map_gt'] = contact_map_gt.reshape(B, self.contact_num, 3)
            ret_dict['contact_map_ori'] = contact_map_ori[0]
            ret_dict['hand_pts'] = contact_map_ori[1].reshape(B, 1024, 3)

        return ret_dict, all_x_t
    
    @torch.no_grad()
    def p_sample_obj2hand(self, data_dict):
        assert self.gen_hand, 'gen_hand must be true'

        hand_param_latent_gen = data_dict['hand_param_latent_gen'].clone()
        hand_R_gen = data_dict['hand_R_gen'].clone()
        hand_t_gen = data_dict['hand_t_gen'].clone()

        pc_global_latent_gen = data_dict['pc_global_latent']
        pc_local_latent_gen = data_dict['pc_local_latent']
        
        B = hand_param_latent_gen.shape[0]

        x_t_dict = {
            'hand_param': hand_param_latent_gen.clone(),
            'hand_R': hand_R_gen.clone(),
            'hand_t': hand_t_gen.clone(),
            'pc_global': pc_global_latent_gen.clone(),
            'pc_local': pc_local_latent_gen.clone(),
            }
        x_t_dict['contact_map'] = None
        if self.gen_contact:
            x_t_dict['contact_map'] = data_dict['contact_map_gen']
                
        all_x_t = [x_t_dict]

        for t in reversed(range(1, self.n_timestep+1)):
            x_t_dict['pc_global'] = pc_global_latent_gen.clone()
            x_t_dict['pc_local'] = pc_local_latent_gen.clone()
            t_hand = torch.ones(B, dtype=torch.long, device=self.device) * t
            t_pc = torch.zeros(B, dtype=torch.long, device=self.device)
            if self.gen_contact:
                t_contact = t_hand
            else:
                t_contact = None
            x_t_new = self.p_sample(x_t_dict, t_hand=t_hand, t_pc=t_pc, t_contact=t_contact)
            all_x_t.append(x_t_new)
            x_t_dict = x_t_new

        final_dict = all_x_t[-1]
        hand_param_final = self.decode_hand_vae(final_dict['hand_param'])

        hand_param_gen_list = []
        hand_t_gen_list = []
        hand_R_gen_list = []
        for d in all_x_t:
            hand_R_gen_list.append(d['hand_R'])
            hand_t_gen_list.append(d['hand_t'])
            hand_param_gen_list.append(self.decode_hand_vae(d['hand_param']))

        pc_final = self.decode_pc_vae(pc_global_latent_gen, pc_local_latent_gen)
        ret_dict = {
            'pc_final': pc_final,
            'hand_param_undecoded': final_dict['hand_param'],
            'hand_param_final': hand_param_final,
            'hand_R_final': final_dict['hand_R'],
            'hand_t_final': final_dict['hand_t'],
            'hand_param_gen_list': hand_param_gen_list,
            'hand_t_gen_list': hand_t_gen_list,
            "hand_R_gen_list": hand_R_gen_list,
            'pc_gen_list': []
        }
        if self.gen_contact:
            contact_map_gen_list = []
            for d in all_x_t:
                contact_map_gen_list.append(d['contact_map'])
            ret_dict["contact_map_final"] = final_dict['contact_map'].reshape(B, self.contact_num, 3)
            ret_dict["contact_map_gen_list"] = contact_map_gen_list
            
            hand_pose = compose_hand_param(final_dict['hand_t'], final_dict['hand_R'], hand_param_final, rot_type=self.hand_rot_type)
            contact_map_gt, contact_map_ori = self.compute_contact_map(dict(object_pc=pc_final, ori_hand_pose=hand_pose))
            ret_dict['contact_map_gt'] = contact_map_gt.reshape(B, self.contact_num, 3)
            ret_dict['contact_map_ori'] = contact_map_ori[0]
            ret_dict['hand_pts'] = contact_map_ori[1].reshape(B, 1024, 3)
                
        return ret_dict, all_x_t

    @torch.no_grad()
    def p_sample_hand2obj(self, data_dict):
        assert self.gen_pc, 'gen_pc must be true'

        pc_global_latent_gen = data_dict['pc_global_latent_gen'].clone()
        pc_local_latent_gen = data_dict['pc_local_latent_gen'].clone()

        B = pc_global_latent_gen.shape[0]

        hand_pose = data_dict['hand_pose'].clone()
        hand_param_latent_gen = data_dict['hand_param_latent']
        if len(hand_pose.shape) == 1:
            hand_pose = hand_pose.unsqueeze(0).repeat(B, 1)
            hand_param_latent_gen = hand_param_latent_gen.unsqueeze(0).repeat(B, 1)
        _, Dh = hand_param_latent_gen.shape

        hand_R_gen = hand_pose[:, self.hand_rot_slice]  # [B, 4]
        hand_t_gen = hand_pose[:, self.hand_trans_slice]  # [B, 3]
        
        x_t_dict = {
            'hand_param': hand_param_latent_gen,
            'hand_R': hand_R_gen,
            'hand_t': hand_t_gen,
            'pc_global': pc_global_latent_gen,
            'pc_local': pc_local_latent_gen,
            }
        
        x_t_dict['contact_map'] = None
        if self.gen_contact:
            x_t_dict['contact_map'] = data_dict['contact_map_gen']
            
        all_x_t = [x_t_dict]

        for t in reversed(range(1, self.n_timestep+1)):
            x_t_dict['hand_param'] = hand_param_latent_gen
            x_t_dict['hand_R'] = hand_R_gen
            x_t_dict['hand_t'] = hand_t_gen
            t_pc = torch.ones(B, dtype=torch.long, device=self.device) * t
            t_hand = torch.zeros(B, dtype=torch.long, device=self.device)
            if self.gen_contact:
                t_contact = t_pc
            else:
                t_contact = None
            x_t_new = self.p_sample(x_t_dict, t_hand=t_hand, t_pc=t_pc, t_contact=t_contact)
            all_x_t.append(x_t_new)
            x_t_dict = x_t_new

        final_dict = all_x_t[-1]
        pc_final = self.decode_pc_vae(final_dict['pc_global'], final_dict['pc_local'])

        pc_gen_list = []
        for d in all_x_t:
            pc_gen_list.append(self.decode_pc_vae(d['pc_global'], d['pc_local']))

        hand_param_final = self.decode_hand_vae(hand_param_latent_gen)
        ret_dict = {
            'pc_final': pc_final,
            'hand_param_final': hand_param_final,
            'hand_R_final': hand_R_gen,
            'hand_t_final': hand_t_gen,
            'hand_param_gen_list': [],
            'hand_t_gen_list': [],
            "hand_R_gen_list": [],
            'pc_gen_list': pc_gen_list,
            'pc_ori': data_dict.get('object_pc', None),
        }
        if self.gen_contact:
            contact_map_gen_list = []
            for d in all_x_t:
                contact_map_gen_list.append(d['contact_map'])
            ret_dict["contact_map_final"] = final_dict['contact_map'].reshape(B, self.contact_num, 3)
            ret_dict["contact_map_gen_list"] = contact_map_gen_list
            
            ori_hand_pose = data_dict['ori_hand_pose']
            contact_map_gt, contact_map_ori = self.compute_contact_map(dict(object_pc=pc_final, ori_hand_pose=ori_hand_pose))
            ret_dict['contact_map_gt'] = contact_map_gt.reshape(B, self.contact_num, 3)
            ret_dict['contact_map_ori'] = contact_map_ori[0]
            ret_dict['hand_pts'] = contact_map_ori[1].reshape(B, 1024, 3)

        return ret_dict, all_x_t

    @torch.no_grad()
    def test_step(self, data_dict, task=['obj2hand']):
        data_dict = self.on_after_batch_transfer(data_dict, 0)
        ret_dict = {}
        if 'joint' in task:
            sample_dict, all_x_t = self.p_sample_joint(data_dict)
            sample_dict['pc_actual'] = self.normalize_back_pcd(sample_dict['pc_final'])
            hand_param_total = compose_hand_param(sample_dict['hand_t_final'], sample_dict['hand_R_final'], 
                                                  sample_dict['hand_param_final'], rot_type=self.hand_rot_type)
            sample_dict['hand_param_composed'] = hand_param_total
            if self.gen_contact:
                sample_dict['contact_map_actual'] = self.normalize_back_pcd(sample_dict['contact_map_final'])
            ret_dict['joint'] = sample_dict
        if 'obj2hand' in task:
            sample_dict, all_x_t = self.p_sample_obj2hand(data_dict)
            hand_param_total = compose_hand_param(sample_dict['hand_t_final'], sample_dict['hand_R_final'], 
                                                  sample_dict['hand_param_final'], rot_type=self.hand_rot_type)
            sample_dict['hand_param_composed'] = hand_param_total
            if self.gen_contact:
                sample_dict['contact_map_actual'] = self.normalize_back_pcd(sample_dict['contact_map_final'])
            ret_dict['obj2hand'] = sample_dict
        if 'hand2obj' in task: 
            sample_dict, all_x_t = self.p_sample_hand2obj(data_dict)
            sample_dict['pc_actual'] = self.normalize_back_pcd(sample_dict['pc_final'])
            hand_param_total = compose_hand_param(sample_dict['hand_t_final'], sample_dict['hand_R_final'], 
                                                  sample_dict['hand_param_final'], rot_type=self.hand_rot_type)
            sample_dict['hand_param_composed'] = hand_param_total
            if self.gen_contact:
                sample_dict['contact_map_actual'] = self.normalize_back_pcd(sample_dict['contact_map_final'])
            ret_dict['hand2obj'] = sample_dict
        return ret_dict
        

    @torch.no_grad()
    def sample(self, data_dict, num=10, task=['joint']):
        data_dict = extract_batch_num(data_dict, num)
        data_dict = dict_to_device(data_dict, self.device)
        data_dict = self.on_after_batch_transfer(data_dict, 0)
        gen_dict = {k: v[:num].to(self.device) for k, v in self.gen_dict.items()}
        if self.gen_contact:
            contact_map, ori_contact_map = self.compute_contact_map(data_dict)
            data_dict['contact_map'] = contact_map
            data_dict['contact_map_ori'] = ori_contact_map[1]
        data_dict.update(gen_dict)
        ret_dict = {}
        if 'joint' in task:
            sample_dict, all_x_t = self.p_sample_joint(data_dict)
            ret_dict['joint'] = sample_dict
        if 'obj2hand' in task:
            sample_dict, all_x_t = self.p_sample_obj2hand(data_dict)
            ret_dict['obj2hand'] = sample_dict
        if 'hand2obj' in task: 
            sample_dict, all_x_t = self.p_sample_hand2obj(data_dict)
            ret_dict['hand2obj'] = sample_dict
        return ret_dict
    
    def sample_vis(self, data_loader, mode, current_save_path):
        num = self.cfg.TRAIN.VAL_SAMPLE_VIS
        batch = next(iter(data_loader))
        batch_dict = dict()
        for k,v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_dict[k] = v[:num].to(self.device)
            elif isinstance(v, list):
                if isinstance(v[0], torch.Tensor):
                    batch_dict[k] = [t.to(self.device) for t in v[:num]]
                else:
                    batch_dict[k] = [t for t in v[:num]]
            else:
                batch_dict[k] = v
        ret_dict = self.sample(data_dict=batch_dict, num=num, task=self.task)
        for task in ret_dict.keys():
            vis_dict = ret_dict[task]
            task_save_path = os.path.join(current_save_path, f"{task}_{mode}")
            if not os.path.exists(task_save_path):
                os.makedirs(task_save_path, exist_ok=True)
            pc = vis_dict['pc_final'].detach().cpu().numpy()
            hand_t = vis_dict['hand_t_final'].detach().cpu().numpy()
            hand_R = vis_dict['hand_R_final'].detach().cpu().numpy()
            hand_param = vis_dict['hand_param_final'].detach().cpu().numpy()
            if "contact_map_final" in vis_dict and vis_dict['contact_map_final'] is not None:
                contact_map = vis_dict['contact_map_final'].detach()
                contact_map_ori = vis_dict['contact_map_ori'].detach().cpu().numpy()
                contact_map = contact_map.cpu().numpy()
            else:
                contact_map = None

            if 'pc_ori' in vis_dict and vis_dict['pc_ori'] is not None:
                pc_ori = vis_dict['pc_ori'].detach().cpu().numpy()
            else:
                pc_ori = None

            hand_param_total = compose_hand_param(hand_t, hand_R, hand_param, rot_type=self.hand_rot_type)

            # save final
            for b in range(num):
                pc_b = pc[b]
                pc_b = pc_b[:, :3]
                xyz = o3d.geometry.PointCloud()
                xyz.points = o3d.utility.Vector3dVector(pc_b)
                if contact_map is not None:
                    pc_new = np.concatenate([pc_b, contact_map[b]], axis=0)
                    color_contact = np.ones_like(contact_map[b])
                    color_contact[:, 0] = 0
                    xyz.points = o3d.utility.Vector3dVector(pc_new)

                    c = 0.5 + 0.5 * contact_map_ori[b]
                    c = c.reshape(-1, 1)
                    color = c * np.array([1., 0., 0.]) + (1. - c) * np.array([1., 1., 1.])
                    color = np.clip(color, 0, 1.)

                    color = np.concatenate([color, color_contact], axis=0)
                    xyz.colors = o3d.utility.Vector3dVector(color)
                else:
                    pc_new = pc_b
                    color = np.ones_like(pc_b)
                    xyz.colors = o3d.utility.Vector3dVector(color)

                if pc_ori is not None:
                    pc_new = np.concatenate([pc_ori[b], pc_new], axis=0)
                    color_ori = np.ones_like(pc_ori[b])
                    color_ori[:, 2] = 0
                    color = np.concatenate([color_ori, color], axis=0)
                    xyz.points = o3d.utility.Vector3dVector(pc_new)
                    xyz.colors = o3d.utility.Vector3dVector(color)

                o3d.io.write_point_cloud(os.path.join(task_save_path, f"pc_{b:03d}.ply"), xyz)
                np.savez(os.path.join(task_save_path, f"hand_{b:03d}.npz"), hand_param=hand_param_total[b], grasp_code=batch_dict['grasp_code'][b])

            # save steps: not yet used
            # if 'hand_param_gen_list' in vis_dict:
                if task == 'joint_partial':
                    continue
                pc_gen_list = vis_dict['pc_gen_list']
                pc_gen_list = [t.detach().cpu().numpy() for t in pc_gen_list]
                hand_t_gen_list = vis_dict['hand_t_gen_list']
                hand_R_gen_list = vis_dict['hand_R_gen_list']
                hand_param_gen_list = vis_dict['hand_param_gen_list']
                hand_param_total_gen_list = []
                for hand_t, hand_R, hand_param in \
                    zip(hand_t_gen_list, hand_R_gen_list, hand_param_gen_list):
                    hand_t = hand_t.detach().cpu().numpy()
                    # print('t max, min', np.max(hand_t), np.min(hand_t))
                    hand_R = hand_R.detach().cpu().numpy()
                    hand_param = hand_param.detach().cpu().numpy()
                    hand_param_t = compose_hand_param(hand_t, hand_R, hand_param, rot_type=self.hand_rot_type)
                    hand_param_total_gen_list.append(hand_param_t)
                if "contact_map_gen_list" in vis_dict:
                    contact_map_gen_list = vis_dict['contact_map_gen_list']
                    contact_map_gen_list = [t.detach().cpu().numpy() for t in contact_map_gen_list]
                    # for t in contact_map_gen_list:
                    #     print('contact max, min', np.max(t), np.min(t))
                else:
                    contact_map_gen_list = []
                for b in range(num):
                    gen_dict = {
                        'pc': [],
                        'hand_param': [],
                        'contact_map': [],
                        'grasp_code': batch_dict['grasp_code'][b],
                    }
                    for stp in range(len(hand_param_total_gen_list)):
                        if self.gen_pc and len(pc_gen_list):
                            gen_dict['pc'].append(pc_gen_list[stp][b])
                        if self.gen_hand and len(hand_param_total_gen_list):
                            gen_dict['hand_param'].append(hand_param_total_gen_list[stp][b])
                        if self.gen_contact and len(contact_map_gen_list):
                            gen_dict['contact_map'].append(contact_map_gen_list[stp][b])
                    with open(os.path.join(task_save_path, f"gen_dict_{b:03d}.pk"), "wb") as f:
                        pickle.dump(gen_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    def run_sample(self, train_loader, test_loader):
        was_training = self.training
        self.eval()
        current_save_path = os.path.join(self.cfg.OUTPUT_PATH, 'eval_save', 'test_043')
        os.makedirs(current_save_path, exist_ok=True)
        self.sample_vis(train_loader, 'train', current_save_path)
        self.sample_vis(test_loader, 'val', current_save_path)
        if self.use_diffusion_optim:
            self.use_diffusion_optim = False
            self.sample_vis(train_loader, 'train_ori', current_save_path)
            self.sample_vis(test_loader, 'val_ori', current_save_path)
        self.training = was_training


if __name__ == '__main__':
    from datetime import datetime

    from dataset import build_dataloader
    from utils.config import cfg
    from utils.dup_stdout_manager import DupStdoutFileManager
    from utils.parse_args import parse_args
    from utils.print_easydict import print_easydict
    args = parse_args("Diffusion")

    torch.manual_seed(cfg.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)

    NOW_TIME = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    file_suffix = NOW_TIME
    if cfg.LOG_FILE_NAME is not None and len(cfg.LOG_FILE_NAME) > 0:
        file_suffix += "_{}".format(cfg.LOG_FILE_NAME)
    full_log_name = f"optim_eval_log_{file_suffix}"
    
    with DupStdoutFileManager(os.path.join(cfg.OUTPUT_PATH, f"{full_log_name}.log")) as _:

        print_easydict(cfg)
        train_loader, val_loader = build_dataloader(cfg)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("using device", device)

        model_save_path = cfg.MODEL_SAVE_PATH
        ckp_files = os.listdir(model_save_path)
        ckp_files = [
            ckp for ckp in ckp_files if ("model_" in ckp) or ("last" in ckp)
        ]
        if len(cfg.WEIGHT_FILE):  # if specify a weight file, load it
            # check if it has training states, or just a model weight
            ckp = torch.load(cfg.WEIGHT_FILE, map_location="cpu")
            # if it has, then it's a checkpoint compatible with pl
            if "state_dict" in ckp.keys():
                ckp_path = cfg.WEIGHT_FILE
            # if it's just a weight, then manually load it to the model
            else:
                ckp_path = None
        elif ckp_files:  # if not specify a weight file, we will check it for you
            ckp_files = sorted(
                ckp_files,
                key=lambda x: os.path.getmtime(os.path.join(model_save_path, x)),
            )
            last_ckp = ckp_files[-1]
            print(f"INFO: automatically detect checkpoint {last_ckp}")
            ckp_path = os.path.join(model_save_path, last_ckp)
        else:
            ckp_path = None

        print(f"ckp_path: {ckp_path}")
        model = UGGGenerationTester(cfg, ckpt=ckp_path, device=device)

        model = model.to(device)

        print("finish setting -----")
        model.run_sample(train_loader, val_loader)

        print("Done test.")


