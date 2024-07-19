from tkinter import E
import numpy as np
import torch
from easydict import EasyDict as edict
from torch import Tensor

from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.utils import square_distance


def cal_pen(hand_model: HandModel, object_pc: torch.Tensor):
    distances = hand_model.cal_distance(object_pc)
    distances[distances <= 0] = 0
    E_pen = distances.sum(-1)
    return E_pen


def cal_joints(hand_model: HandModel):
    # E_joints
    E_joints = torch.sum((hand_model.hand_pose[:, 9:] > hand_model.joints_upper) * (hand_model.hand_pose[:, 9:] - hand_model.joints_upper), dim=-1) + \
        torch.sum((hand_model.hand_pose[:, 9:] < hand_model.joints_lower) * (hand_model.joints_lower - hand_model.hand_pose[:, 9:]), dim=-1)
    return E_joints


def cal_spen(hand_model: HandModel):
    E_spen = hand_model.self_penetration()
    return E_spen


def cal_contact(hand_model: HandModel, contact_points: torch.Tensor):
    # E_contact
    distances_contact = hand_model.cal_distance(contact_points)
    E_contact = distances_contact.abs().sum(-1)
    return E_contact


def cal_contact_pc(pc: Tensor, contact_points: Tensor):
    # E_contact
    distances_contact = square_distance(contact_points, pc)
    E_contact_pc = distances_contact.abs().min(-2)[0]
    E_contact_pc = E_contact_pc.sum(-1)
    return E_contact_pc


def cal_dis_fc(hand_model: HandModel, object_model: ObjectModel):
    # Requires object_model (the SDF of the object),
    # not supposed to be used for testing
    device = hand_model.device
    distance_, contact_normal_ = object_model.cal_distance(hand_model.contact_points)  # [B, 140], [B, 140, 3]
    distance, contact_idx = distance_.topk(k=5, dim=-1)  # largest 4 points (closer bigger) [B, 4], [B, 4]
    contact_normal = torch.gather(contact_normal_, dim=1, index=contact_idx
                                    .unsqueeze(-1).repeat(1, 1, 3)) # [B, 4, 3]
    B, N_contact = distance.shape
    hand_model.set_contact_point_indices(contact_idx)
    E_dis = torch.sum(distance.abs(), dim=-1, dtype=torch.float)
    # E_fc
    contact_normal = contact_normal.reshape(B, 1, 3 * N_contact).float()
    transformation_matrix = torch.tensor([[0, 0, 0, 0, 0, -1, 0, 1, 0],
                                        [0, 0, 1, 0, 0, 0, -1, 0, 0],
                                        [0, -1, 0, 1, 0, 0, 0, 0, 0]],
                                        dtype=torch.float, device=device)
    g = torch.cat([torch.eye(3, dtype=torch.float, device=device)
                   .expand(B, N_contact, 3, 3).reshape(B, 3 * N_contact, 3),
                   (hand_model.contact_points @ transformation_matrix).view(B, 3 * N_contact, 3)], 
                dim=2).float().to(device)
    norm = torch.norm(contact_normal @ g, dim=[1, 2])
    E_fc = norm * norm
    return E_dis, E_fc


def cal_tta_energy_hand_pose(hand_model: HandModel, object_model: ObjectModel, contact_points: Tensor, weight_dict=edict()):
    E_contact = cal_contact(hand_model, contact_points)
    E_pen = cal_pen(hand_model, object_model.object_pc)
    E_spen = cal_spen(hand_model)
    E_joints = cal_joints(hand_model)
    loss = weight_dict.w_contact * E_contact + weight_dict.w_pen * E_pen + weight_dict.w_spen * E_spen + weight_dict.w_joints * E_joints
    losses = dict(E_dis=E_contact, E_pen=E_pen, E_spen=E_spen, E_joints=E_joints)
    return loss, losses
    

def cal_tta_energy_pc(hand_model: HandModel, object_pc: Tensor, contact_points: Tensor, weight_dict=edict()):
    E_contact = cal_contact(hand_model, contact_points)
    E_contact_pc = cal_contact_pc(object_pc, contact_points)
    E_pen = cal_pen(hand_model, object_pc)
    loss = weight_dict.w_contact * E_contact + weight_dict.w_contact * E_contact_pc + weight_dict.w_pen * E_pen
    losses = dict(E_contact=E_contact, E_contact_pc=E_contact_pc, E_pen=E_pen)
    return loss, losses

def cal_tta_energy_joint(hand_model: HandModel, object_pc: Tensor, contact_points: Tensor, weight_dict=edict()):
    E_contact = cal_contact(hand_model, contact_points)
    E_contact_pc = cal_contact_pc(object_pc, contact_points)
    E_pen = cal_pen(hand_model, object_pc)
    E_spen = cal_spen(hand_model)
    E_joints = cal_joints(hand_model)
    loss = weight_dict.w_contact * E_contact + weight_dict.w_contact * E_contact_pc + weight_dict.w_pen * E_pen + weight_dict.w_spen * E_spen + weight_dict.w_joints * E_joints
    losses = dict(E_dis=E_contact, E_contact_pc=E_contact_pc, E_pen=E_pen, E_spen=E_spen, E_joints=E_joints)
    return loss, losses
