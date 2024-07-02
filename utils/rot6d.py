import torch
import numpy as np
import torch.nn.functional as F


def compute_rotation_matrix_from_ortho6d(poses):
    """
    Code from
    https://github.com/papagina/RotationContinuity
    On the Continuity of Rotation Representations in Neural Networks
    Zhou et al. CVPR19
    https://zhouyisjtu.github.io/project_rotation/rotation.html
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3
        
    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3
        
    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix

def robust_compute_rotation_matrix_from_ortho6d(poses):
    """
    Instead of making 2nd vector orthogonal to first
    create a base that takes into account the two predicted
    directions equally
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    y = normalize_vector(y_raw)  # batch*3
    middle = normalize_vector(x + y)
    orthmid = normalize_vector(x - y)
    x = normalize_vector(middle + orthmid)
    y = normalize_vector(middle - orthmid)
    # Their scalar product should be small !
    # assert torch.einsum("ij,ij->i", [x, y]).abs().max() < 0.00001
    z = normalize_vector(cross_product(x, y))

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    # Check for reflection in matrix ! If found, flip last vector TODO
    # assert (torch.stack([torch.det(mat) for mat in matrix ])< 0).sum() == 0
    return matrix


def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, v.new([1e-8]))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v/v_mag
    return v


def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        
    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)
        
    return out


def normalize_rot6d_torch(rot):
    if rot.shape[-1] == 3:
        unflatten = True
        rot = rot.flatten(-2, -1)
    else:
        unflatten = False
    a1, a2 = rot[..., :3], rot[..., 3:]
    b1 = F.normalize(a1, p=2, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, p=2, dim=-1)
    rot = torch.cat([b1, b2], dim=-1)  # back to [..., 6]
    if unflatten:
        rot = rot.unflatten(-1, (2, 3))
    return rot


def normalize_np(x):
    x_n = np.linalg.norm(x, axis=-1, keepdims=True)
    x_n = x_n.clip(min=1e-8)
    x = x / x_n
    return x


def normalize_rot6d_numpy(rot):
    if rot.shape[-1] == 3:
        unflatten = True
        undim = True
        ori_shape = rot.shape[:-2]
        p = np.prod(ori_shape)
        rot = rot.reshape(p, 6)
    elif len(rot.shape) > 2:
        unflatten = False
        undim = True
        ori_shape = rot.shape[:-1]
        p = np.prod(ori_shape)
        rot = rot.reshape(p, 6)
    else:
        unflatten = False
        undim = False
        ori_shape = None
    a1, a2 = rot[:, :3], rot[:, 3:]
    b1 = normalize_np(a1)
    b2 = a2 - (b1 * a2).sum(axis=-1, keepdims=True) * b1
    b2 = normalize_np(b2)
    rot = np.concatenate([b1, b2], axis=-1)  # back to [..., 6]
    if unflatten:
        rot = rot.reshape(ori_shape + (2, 3))
    elif undim:
        rot = rot.reshape(ori_shape + (6, ))
    return rot

def normalize_rot6d(rot):
    if isinstance(rot, torch.Tensor):
        return normalize_rot6d_torch(rot)
    elif isinstance(rot, np.ndarray):
        return normalize_rot6d_numpy(rot)
    else:
        raise NotImplementedError
    