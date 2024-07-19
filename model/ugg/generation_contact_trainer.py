import numpy as np
import torch
import torch.nn as nn
import yaml
from easydict import EasyDict as edict
from loguru import logger

import LION.vae_adain as vae_adain
from model.modules.base_lightning import BaseLightningModel
from utils.distribution import sample_normal
from utils.hand_helper import ROT_DIM_DICT, compose_hand_param
from utils.hand_model import HandModel
from utils.utils import dict_to_device, extract_batch_num

from .diffusion_utils.ddpm_schedule import (DDPMSchedule,
                                            diffusion_beta_schedule)
from .hand_vae import HandTrainer, HandVAE
from .UViT import UViTContact

LATENT_NAME_LIST = ['pc_global', 'pc_local', 'hand_param', 'hand_R', 'hand_t', 'contact_map']

class UGGGenerationTrainer(BaseLightningModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_cfg = cfg.MODEL
        self.diffusion_cfg = cfg.MODEL.diffusion
        
        self.apply_random_rot = cfg.DATA.APPLY_RANDOM_ROT
        print("apply random rotation", self.apply_random_rot)

        with open(self.diffusion_cfg.pc_args, 'r') as f:
            pc_args = edict(yaml.full_load(f))
        self.pc_latent_model = vae_adain.Model(cfg, pc_args)

        if len(self.diffusion_cfg.pc_checkpoint) and self.model_cfg.diffusion.pc_checkpoint.lower() != 'none':
            # if has pretrained ckpt, we don't need to load the vae ckpt anymore
            print('Load vae_checkpoint: {}', self.model_cfg.diffusion.pc_checkpoint)
            vae_ckpt = torch.load(self.model_cfg.diffusion.pc_checkpoint)
            vae_weight = vae_ckpt['model']
            self.pc_latent_model.load_state_dict(vae_weight)
            for name, para in self.pc_latent_model.named_parameters():
                para.requires_grad = False
        elif len(self.diffusion_cfg.pc_checkpoint) == 0 and len(self.cfg.WEIGHT_FILE) == 0:
            assert False, 'should have checkpoint for pc encoder'
        
        self.pc_global_dim = self.diffusion_cfg.pc_global_dim
        self.hand_param_dim = self.diffusion_cfg.hand_param_dim
        self.pc_local_pts = self.diffusion_cfg.pc_local_pts
        self.pc_local_pts_dim = self.diffusion_cfg.pc_local_pts_dim
        self.contact_num = 5
        
        self.encode_hand = self.diffusion_cfg.encode_hand
        if self.encode_hand:
            self.hand_latent_model = HandVAE(cfg)
            if len(self.model_cfg.diffusion.hand_checkpoint) and self.model_cfg.diffusion.hand_checkpoint.lower()!= 'none':
                print('Load hand_checkpoint: {}', self.model_cfg.diffusion.hand_checkpoint)
                hand_trainer = HandTrainer.load_from_checkpoint(self.model_cfg.diffusion.hand_checkpoint)
                self.hand_latent_model = hand_trainer.hand_model
                for name, para in self.hand_latent_model.named_parameters():
                    para.requires_grad = False
            elif len(self.diffusion_cfg.hand_checkpoint) == 0 and len(self.cfg.WEIGHT_FILE) == 0:
                assert False, 'should have checkpoint for hand encoder'
        else:
            self.hand_latent_model = None

        self._set_hand_rot()

        self.unet = UViTContact(cfg)
        
        self.n_timestep = self.diffusion_cfg.n_timestep
        _betas = diffusion_beta_schedule(start=self.diffusion_cfg.beta_start,
                                         end=self.diffusion_cfg.beta_end,
                                         n_timestep=self.diffusion_cfg.n_timestep)
        self.schedule = DDPMSchedule(_betas)

        self.criterion_hand = self._set_loss(self.diffusion_cfg.loss_hand)
        self.criterion_pc = self._set_loss(self.diffusion_cfg.loss_pc)
        self.criterion_contact = self._set_loss(self.diffusion_cfg.loss_contact)

        self.contact_map_normalize_factor = self.diffusion_cfg.contact_map_normalize_factor
        self.normalize_pc = self.cfg.DATA.NORMALIZE_PC

        self._prep_generation()

        self._set_gen()

        self._set_fixed_param()

        self.hand_model = None
    
    def _set_fixed_param(self):
        for name, para in self.pc_latent_model.named_parameters():
            para.requires_grad = False
        if self.hand_latent_model is not None:
            for name, para in self.hand_latent_model.named_parameters():
                para.requires_grad = False

    def _set_loss(self, loss):
        if loss == 'l2':
            return nn.MSELoss()
        elif loss == 'l1':
            return nn.L1Loss()
        elif loss == 'bce':
            return nn.BCELoss()
        else:
            raise NotImplementedError(f"loss {loss} not implemented")

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
        if self.apply_random_rot:
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
        else:
            obj_pc = batch['object_pc']
            pc_vae_output = self.compute_pc_vae(obj_pc)
            pc_global_latent = pc_vae_output['eps_global']  # [B, D1]
            pc_local_latent = pc_vae_output['eps_local']  # [B, ND2]
            
        # compute hand parameter latent
        hand_pose = batch['hand_pose']
        hand_param = hand_pose[:, self.hand_param_slice]
        hand_vae_output = self.compute_hand_vae(hand_param)
        hand_param_latent = hand_vae_output['eps']
        
        batch.update({
            'pc_global_latent': pc_global_latent,
            'pc_local_latent': pc_local_latent,
            'hand_param_latent': hand_param_latent,
        })
        
        # compute contact map
        if self.gen_contact:
            contact_map, ori_contact_map = self.compute_contact_map(batch)
            batch['contact_map'] = contact_map
        return batch
        
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
        if self.normalize_pc:
            pc = pc / 6.6
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
        if self.normalize_pc:
            contact_map *= 6.6

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
        hand_pose = data_dict['hand_pose']
        B = hand_pose.shape[0]
        
        pc_global_latent = data_dict['pc_global_latent']
        pc_local_latent = data_dict['pc_local_latent']
        
        _, D1 = pc_global_latent.shape
        _, ND2 = pc_local_latent.shape
        if self.gen_pc:
            pc_latent = torch.cat([pc_global_latent, pc_local_latent], dim=-1)
            t_pc, eps_pc, xn_pc = self.schedule.sample(pc_latent)

            pc_global_eps = eps_pc[:, :D1]
            pc_local_eps = eps_pc[:, D1:]

            pc_global_xn = xn_pc[:, :D1]
            pc_local_xn = xn_pc[:, D1:]
        else:
            t_pc = torch.zeros(B, dtype=torch.long, device=self.device)
            pc_global_eps = pc_local_eps = None
            pc_global_xn = pc_global_latent
            pc_local_xn = pc_local_latent

        hand_param_latent = data_dict['hand_param_latent']
        _, Dh = hand_param_latent.shape

        hand_R = hand_pose[:, self.hand_rot_slice]  # [B, 4]
        hand_t = hand_pose[:, self.hand_trans_slice]  # [B, 3]

        if self.gen_hand:
            hand_latent = torch.cat([hand_t, hand_R, hand_param_latent], dim=-1)
            t_hand, eps_hand, xn_hand = self.schedule.sample(hand_latent)

            hand_t_eps = eps_hand[:, self.hand_trans_slice]
            hand_R_eps = eps_hand[:, self.hand_rot_slice]
            hand_param_eps = eps_hand[:, self.hand_param_slice]

            hand_t_xn = xn_hand[:, self.hand_trans_slice]
            hand_R_xn = xn_hand[:, self.hand_rot_slice]
            hand_param_xn = xn_hand[:, self.hand_param_slice]
        else:
            t_hand = torch.zeros(B, dtype=torch.long, device=self.device)
            hand_t_eps = hand_R_eps = hand_param_eps = None
            hand_t_xn = hand_t
            hand_R_xn = hand_R
            hand_param_xn = hand_param_latent

        unet_dict = {
            'pc_global': pc_global_xn,
            'pc_local': pc_local_xn,
            'hand_param': hand_param_xn,
            'hand_R': hand_R_xn,
            'hand_t': hand_t_xn,
            't_pc': t_pc,
            't_hand': t_hand
        }
        
        if self.gen_contact:
            contact_map = data_dict['contact_map']
            contact_map = contact_map.reshape(B, -1)
            t_contact, eps_contact_map, xn_contact = self.schedule.sample(contact_map)
            unet_dict.update({
                'contact_map': xn_contact,
                't_contact': t_contact,
            })

        unet_output = self.unet(unet_dict)

        gt_output = {
            'hand_t_eps': hand_t_eps,
            'hand_R_eps': hand_R_eps,
            'hand_param_eps': hand_param_eps,
            'pc_global_eps': pc_global_eps,
            'pc_local_eps': pc_local_eps,
            'hand_param_latent': hand_param_latent,
            'pc_global_latent': pc_global_latent,
            'pc_local_latent': pc_local_latent,
            'batch_size': B,
        }

        if self.gen_contact:
            gt_output.update({
                'contact_map_eps': eps_contact_map,
                'contact_map': contact_map,
            })

        gt_output.update(unet_output)
        
        return gt_output
    
    def loss_function(self, data_dict, out_dict, optimizer_idx=-1):
        loss = torch.zeros(1, device=self.device)
        
        loss_dict = {}
        if self.gen_hand:
            hand_param_eps_pred = out_dict['hand_param_eps_pred']
            hand_param_eps = out_dict['hand_param_eps']
            hand_R_eps_pred = out_dict['hand_R_eps_pred']
            hand_R_eps = out_dict['hand_R_eps']
            hand_t_eps_pred = out_dict['hand_t_eps_pred']
            hand_t_eps = out_dict['hand_t_eps']
            hand_param_loss = self.criterion_hand(hand_param_eps_pred, hand_param_eps)
            hand_R_loss = self.criterion_hand(hand_R_eps_pred, hand_R_eps)
            hand_t_loss = self.criterion_hand(hand_t_eps_pred, hand_t_eps)
            loss += hand_param_loss + hand_R_loss + hand_t_loss

            loss_dict['hand_param_loss'] = hand_param_loss
            loss_dict['hand_R_loss'] = hand_R_loss
            loss_dict['hand_t_loss'] = hand_t_loss
        else:
            hand_param_loss = hand_R_loss = hand_t_loss = torch.zeros(1, device=self.device)

        if self.gen_pc:
            pc_global_eps_pred = out_dict['pc_global_eps_pred']
            pc_global_eps = out_dict['pc_global_eps']
            pc_local_eps_pred = out_dict['pc_local_eps_pred']
            pc_local_eps = out_dict['pc_local_eps']
            pc_global_loss = self.criterion_pc(pc_global_eps_pred, pc_global_eps)
            pc_local_loss = self.criterion_pc(pc_local_eps_pred, pc_local_eps)
            loss += pc_global_loss + pc_local_loss

            loss_dict['pc_local_loss'] = pc_local_loss
            loss_dict['pc_global_loss'] = pc_global_loss
        else:
            pc_global_loss = pc_local_loss = torch.zeros(1, device=self.device)

        if self.gen_contact:
            contact_map_eps_pred = out_dict['contact_map_eps_pred']
            contact_map_eps = out_dict['contact_map_eps']
            contact_map_loss = self.criterion_contact(contact_map_eps_pred, contact_map_eps)
            loss += contact_map_loss

            loss_dict['contact_map_loss'] = contact_map_loss
        else:
            contact_map_loss = torch.zeros(1, device=self.device)

        loss_dict['loss'] = loss
        loss_dict['batch_size'] = out_dict['batch_size']
        return loss_dict
        
    @torch.no_grad()
    def p_sample(self, x_t, t_pc, t_hand, t_contact=None):
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
        hand_param_latent_gen = data_dict['hand_param_latent_gen']
        hand_R_gen = data_dict['hand_R_gen']
        hand_t_gen = data_dict['hand_t_gen']
        pc_global_latent_gen = data_dict['pc_global_latent_gen']
        pc_local_latent_gen = data_dict['pc_local_latent_gen']
        B = hand_param_latent_gen.shape[0]

        x_t_dict = {
            'hand_param': hand_param_latent_gen,
            'hand_R': hand_R_gen,
            'hand_t': hand_t_gen,
            'pc_global': pc_global_latent_gen,
            'pc_local': pc_local_latent_gen,
        }
        if self.gen_contact:
            x_t_dict['contact_map'] = data_dict['contact_map_gen']
        else:
            x_t_dict['contact_map'] = None

        all_x_t = [x_t_dict]

        for t in reversed(range(1, self.n_timestep+1)):
            t_ = torch.ones(B, dtype=torch.long, device=self.device) * t
            if self.gen_contact:
                t_contact = t_
            else:
                t_contact = None
            x_t_new = self.p_sample(x_t_dict, t_hand=t_, t_pc=t_, t_contact=t_contact)
            all_x_t.append(x_t_new)
            x_t_dict = x_t_new

        final_dict = all_x_t[-1]
        hand_param_latent = final_dict['hand_param']

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
        hand_param_latent_gen = data_dict['hand_param_latent_gen']
        hand_R_gen = data_dict['hand_R_gen']
        hand_t_gen = data_dict['hand_t_gen']
        
        pc_global_latent_gen = data_dict['pc_global_latent']
        pc_local_latent_gen = data_dict['pc_local_latent']

        B = hand_param_latent_gen.shape[0]

        x_t_dict = {
            'hand_param': hand_param_latent_gen,
            'hand_R': hand_R_gen,
            'hand_t': hand_t_gen,
            'pc_global': pc_global_latent_gen,
            'pc_local': pc_local_latent_gen,
            }
        if self.gen_contact:
            x_t_dict['contact_map'] = data_dict['contact_map_gen']
        else:
            x_t_dict['contact_map'] = None

        all_x_t = [x_t_dict]

        for t in reversed(range(1, self.n_timestep+1)):
            x_t_dict['pc_global'] = pc_global_latent_gen
            x_t_dict['pc_local'] = pc_local_latent_gen
            t_hand = torch.ones(B, dtype=torch.long, device=self.device) * t
            t_pc = torch.zeros(B, dtype=torch.long, device=self.device)
            if self.gen_contact:
                t_contact = torch.ones(B, dtype=torch.long, device=self.device) * t
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
        pc_global_latent_gen = data_dict['pc_global_latent_gen']
        pc_local_latent_gen = data_dict['pc_local_latent_gen']
        
        hand_pose = data_dict['hand_pose']
        hand_param_latent_gen = data_dict['hand_param_latent']
        _, Dh = hand_param_latent_gen.shape

        hand_R_gen = hand_pose[:, self.hand_rot_slice]  # [B, 4]
        hand_t_gen = hand_pose[:, self.hand_trans_slice]  # [B, 3]

        B = hand_param_latent_gen.shape[0]

        x_t_dict = {
            'hand_param': hand_param_latent_gen,
            'hand_R': hand_R_gen,
            'hand_t': hand_t_gen,
            'pc_global': pc_global_latent_gen,
            'pc_local': pc_local_latent_gen,
            }
        if self.gen_contact:
            x_t_dict['contact_map'] = data_dict['contact_map_gen']
        else:
            x_t_dict['contact_map'] = None

        all_x_t = [x_t_dict]

        for t in reversed(range(1, self.n_timestep+1)):
            x_t_dict['hand_param'] = hand_param_latent_gen
            x_t_dict['hand_R'] = hand_R_gen
            x_t_dict['hand_t'] = hand_t_gen
            t_pc = torch.ones(B, dtype=torch.long, device=self.device) * t
            t_hand = torch.zeros(B, dtype=torch.long, device=self.device)
            if self.gen_contact:
                t_contact = torch.ones(B, dtype=torch.long, device=self.device) * t
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
        }
        if self.gen_contact:
            contact_map_gen_list = []
            for d in all_x_t:
                contact_map_gen_list.append(d['contact_map'])
            ret_dict["contact_map_final"] = final_dict['contact_map'].reshape(B, self.contact_num, 3)
            ret_dict["contact_map_gen_list"] = contact_map_gen_list
            hand_pose = compose_hand_param(hand_t_gen, hand_R_gen, hand_param_final, rot_type=self.hand_rot_type)
            contact_map_gt, contact_map_ori = self.compute_contact_map(dict(object_pc=pc_final, ori_hand_pose=hand_pose))
            ret_dict['contact_map_gt'] = contact_map_gt.reshape(B, self.contact_num, 3)
            ret_dict['contact_map_ori'] = contact_map_ori[0]
            ret_dict['hand_pts'] = contact_map_ori[1].reshape(B, 1024, 3)

        return ret_dict, all_x_t
    
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

    def load_state_dict(self, state_dict,
                        strict: bool = True):
        super().load_state_dict(state_dict, strict=False)
        