import random
from multiprocessing import Pool

import numpy as np
import scipy.spatial
import torch

from .hand_model import HandModel
from .object_model import ObjectModel


def cal(_):
    cfg, contact_points_object, contact_normals = _
    if contact_points_object.shape == ():
        return 0
    n_contact = len(contact_points_object)

    # cal contact forces
    u1 = np.stack([-contact_normals[:, 1], contact_normals[:, 0], np.zeros([n_contact], dtype=np.float32)], axis=1)
    u2 = np.stack([np.ones([n_contact], dtype=np.float32), np.zeros([n_contact], dtype=np.float32), np.zeros([n_contact], dtype=np.float32)], axis=1)
    u = np.where(np.linalg.norm(u1, axis=1, keepdims=True) > 1e-8, u1, u2)
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(u, contact_normals)
    theta = np.linspace(0, 2 * np.pi, cfg.m, endpoint=False).reshape(cfg.m, 1, 1)
    contact_forces = (contact_normals + cfg.mu * (np.cos(theta) * u + np.sin(theta) * v)).reshape(-1, 3)

    # cal wrench space and q1
    origin = np.array([0, 0, 0], dtype=np.float32)
    wrenches = np.concatenate([np.concatenate([contact_forces, cfg.lambda_torque * np.cross(np.tile(contact_points_object - origin, (cfg.m, 1)), contact_forces)], axis=1), np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float32)], axis=0)
    try:
        wrench_space = scipy.spatial.ConvexHull(wrenches)
    except scipy.spatial.qhull.QhullError:
        return 0
    q1 = np.array([1], dtype=np.float32)
    for equation in wrench_space.equations:
        q1 = np.minimum(q1, np.abs(equation[6]) / np.linalg.norm(equation[:6]))
    return q1.item()


def cal_q1(cfg, B, object_model: ObjectModel, hand_model: HandModel, device, with_contact=False):
    distance_, contact_normal_, closest_points_ = object_model.cal_distance(hand_model.contact_points, with_closest_points=True)
    contact_points = [[] for i in range(B)]
    contact_normals = [[] for i in range(B)]
    if cfg.nms:
        nearest_point_index = distance_.argmax(dim=1)
        for num in range(B):
            if - distance_[num, nearest_point_index[num]] < cfg.thres_contact:
                contact_points[num].append(closest_points_[num, nearest_point_index[num]])
                contact_normals[num].append(contact_normal_[num, nearest_point_index[num]])
    else:
        for num in range(B):
            contact_idx = (-distance_[num] < cfg.thres_contact).nonzero().reshape(-1)
            if len(contact_idx) != 0:
                if len(contact_idx) > cfg.max_contact:
                    random.shuffle(contact_idx)
                    contact_idx = contact_idx[:cfg.max_contact]
                for idx in contact_idx:
                    contact_points[num].append(closest_points_[num, idx])
                    contact_normals[num].append(contact_normal_[num, idx])

    for num in range(B):
        if len(contact_points[num]) > 0:
            contact_points[num] = torch.stack(contact_points[num])
            contact_normals[num] = torch.stack(contact_normals[num])
        else:
            contact_points[num] = torch.tensor([[0, 0, 0]], dtype=torch.float, device=device)
            contact_normals[num] = torch.tensor([[1, 0, 0]], dtype=torch.float, device=device)
    params = []
    for i in range(B):
        params.append([cfg, contact_points[i].cpu().numpy(), -contact_normals[i].cpu().numpy()])

    with Pool(cfg.n_cpu) as p:
        q1_list = list(p.map(cal, params))

    if with_contact:
        return torch.tensor(q1_list, dtype=torch.float, device=device), contact_points, contact_normals
    else:
        return torch.tensor(q1_list, dtype=torch.float, device=device)
    

def cal_pen(hand_model: HandModel, pc: torch.Tensor):
    distances = hand_model.cal_distance(pc)
    distances[distances <= 0] = 0
    E_pen = distances.max(dim=1).values
    return E_pen
