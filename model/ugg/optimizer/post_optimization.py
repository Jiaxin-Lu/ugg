import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from pytorch3d.transforms import quaternion_to_matrix
from tqdm import tqdm

from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.utils import dict_to_detach

from .cal_energy import (cal_tta_energy_hand_pose, cal_tta_energy_joint,
                         cal_tta_energy_pc)


def post_optimization_pose(cfg, hand_pose: torch.Tensor, hand_model: HandModel, object_model: ObjectModel, contact_points, steps=100):    
    hand_pose = hand_pose.clone()
    hand_pose.requires_grad_(True)
    optimizer = torch.optim.Adam([hand_pose], lr=cfg.opt_lr)
    loss_list = []
    hand_pose_list = []
    for stp in tqdm(range(steps), desc="Refine:", total=steps, leave=False):
    # for stp in range(steps):
        optimizer.zero_grad()
        hand_model.set_parameters(hand_pose)
        loss, losses = cal_tta_energy_hand_pose(hand_model, object_model, contact_points, cfg)
        loss = loss.mean()

        loss_list.append(dict_to_detach(losses))
        loss.backward()
        optimizer.step()
        hand_pose_list.append(hand_pose.clone().detach())
    hand_pose_final = hand_pose.detach()
    
    return hand_pose_final, loss_list, hand_pose_list


def post_optimization_pc(cfg, pc, hand_pose, hand_model: HandModel, contact_points, steps=100):
    # optimize the hand position and object position
    # no change on pc shape
    # if needed, it is recommended to change pc shape 
    # after this round of optimization
    pc = pc.clone()
    pc_quat = torch.tensor([[1, 0, 0, 0]], dtype=pc.dtype, device=pc.device).repeat(pc.shape[0], 1)
    pc_trans = torch.tensor([[0, 0, 0]], dtype=pc.dtype, device=pc.device).repeat(pc.shape[0], 1)
    pc_quat.requires_grad_(True)
    pc_trans.requires_grad_(True)
    hand_transformation = hand_pose[..., :9]
    hand_param = hand_pose[..., 9:].clone()
    hand_transformation.requires_grad_(True)
    
    optimizer = torch.optim.Adam([pc_quat, pc_trans, hand_transformation], lr=cfg.opt_lr)
    pc_list = []
    hand_pose_list = []
    loss_list = []
    
    for stp in tqdm(range(steps), desc="Refine:", total=steps, leave=False):
        optimizer.zero_grad()
        pc_rot = quaternion_to_matrix(F.normalize(pc_quat, p=2, dim=1))
        pc_in = torch.bmm(pc, pc_rot) + pc_trans[:, None]
        hand_pose_in = torch.cat([hand_transformation, hand_param], dim=-1)
        hand_model.set_parameters(hand_pose_in)
        loss, losses = cal_tta_energy_pc(hand_model, pc_in, contact_points, cfg)

        loss = loss.mean()
        loss_list.append(dict_to_detach(losses))
        loss.backward()
        optimizer.step()
        pc_list.append(pc_in.clone().detach())
        hand_pose_list.append(hand_pose_in.clone().detach())
        
    pc_rot = quaternion_to_matrix(F.normalize(pc_quat, p=2, dim=1))
    pc_final = torch.bmm(pc, pc_rot) + pc_trans[:, None]
    pc_final = pc_final.detach()
    hand_pose_final = torch.cat([hand_transformation, hand_param], dim=-1).detach()
    return pc_final, hand_pose_final, loss_list, pc_list, hand_pose_list


def post_optimization_joint(cfg, hand_pose: torch.Tensor, hand_model: HandModel, pc: torch.Tensor, contact_points=None, steps=100):
    # optimize the hand pose and object position
    # no change on pc shape
    # if needed, it is recommended to change pc shape 
    # after this round of optimization
    hand_pose = hand_pose.clone()
    pc = pc.clone()
    hand_pose.requires_grad_(True)
    pc_quat = torch.tensor([[1, 0, 0, 0]], dtype=pc.dtype, device=pc.device).repeat(pc.shape[0], 1)
    pc_trans = torch.tensor([[0, 0, 0]], dtype=pc.dtype, device=pc.device).repeat(pc.shape[0], 1)
    pc_quat.requires_grad_(True)
    pc_trans.requires_grad_(True)
    
    optimizer = torch.optim.Adam([pc_quat, pc_trans, hand_pose], lr=cfg.opt_lr)
    pc_list = []
    hand_pose_list = []
    loss_list = []
    
    for stp in range(steps):
        optimizer.zero_grad()
        pc_rot = quaternion_to_matrix(F.normalize(pc_quat, p=2, dim=1))
        pc_in = torch.bmm(pc, pc_rot) + pc_trans[:, None]
        hand_model.set_parameters(hand_pose)
        loss, losses = cal_tta_energy_joint(hand_model, pc_in, contact_points, cfg)
        
        loss = loss.mean()
        loss_list.append(dict_to_detach(losses))
        loss.backward()
        optimizer.step()
        pc_list.append(pc_in.clone().detach())
        hand_pose_list.append(hand_pose.clone().detach())
        
    pc_rot = quaternion_to_matrix(F.normalize(pc_quat, p=2, dim=1))
    pc_final = torch.bmm(pc, pc_rot) + pc_trans[:, None]
    pc_final = pc_final.detach()
    hand_pose_final = hand_pose.detach()
    return hand_pose_final, pc_final, loss_list, hand_pose_list, pc_list

