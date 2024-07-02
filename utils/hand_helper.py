import math
import numpy as np
import torch
from .rot6d import normalize_rot6d_numpy, normalize_rot6d_torch

HAND_POSE_MEAN = [ 0.00866105,  0.23923993,  0.5744655,   0.18574114,  0.01515685,  0.2184931,
                0.6651551,  0.19167383,  0.02034698,  0.28905097,  0.65397215,  0.18671101,
                0.2992066,  -0.0690463,   0.24670933,  0.60648596,  0.19446549,  0.30786476,
                1.0769926,   0.00617456, -0.2993749,  -0.1039129, ]
HAND_POSE_STD = [0.15125905, 0.16166152, 0.24705301, 0.21125342, 0.13239177, 0.1387635,
                0.2747734, 0.21673404, 0.1498048, 0.15132234, 0.25578833, 0.20777941,
                0.12084074, 0.14997107, 0.12408759, 0.25629887, 0.2133086, 0.21647978,
                0.12440959, 0.10922779, 0.18252645, 0.14209746]
HAND_POSE_MIN = [-1.3174298,  -0.50528544, -0.29243267, -0.5887583,  -1.1498867,  -0.67438334,
                -0.21203011, -0.80237496, -0.8219279,  -0.4905336,  -0.30009413, -0.40315303,
                -0.3069198,  -0.9611522,  -0.28075632, -0.22816268, -0.47040823, -0.76634,
                0.17260288, -0.7636632,  -1.0075126,  -1.7423708, ]
HAND_POSE_MAX = [1.1163847,  1.2857847,  2.4173303,  2.1739862,  1.0859628,  1.3800644,
                2.5540183,  2.266491,   1.0346404,  1.2404677,  2.172077,   2.21667,
                0.8385511,  1.0727106,  1.2466334,  2.0638,     1.8391026,  1.0431478,
                1.6855251,  0.6073609,  0.5251439,  0.24864246,]

POS_MEAN_AX = [0, 0, 0, 0, 0, 0]
POS_STD_AX = [0.06, 0.06, 0.06, math.pi * 0.2, math.pi * 0.3, math.pi * 0.2]

POS_MEAN_QUAT = [0, 0, 0, 0, 0, 0, 0]
POS_STD_QUAT = [0.06, 0.06, 0.06, 0.2, 0.2, 0.2, 0.2]

T_MEAN = [0, 0, 0]
T_STD = [0.06, 0.06, 0.06]
T_MIN = [-0.3, -0.3, -0.3]
T_MAX = [0.3, 0.3, 0.3]

R_MEAN_QUAT = [0, 0, 0, 0]
R_STD_QUAT = [0.2, 0.2, 0.2, 0.2]
R_MIN_QUAT = [-1, -1, -1, -1]
R_MAX_QUAT = [1, 1, 1, 1]

R_MEAN_AX = [0, 0, 0]
R_STD_AX = [math.pi * 0.2, math.pi * 0.3, math.pi * 0.2]
R_MIN_AX = [-math.pi, -1.5 * math.pi, -math.pi]
R_MAX_AX = [math.pi, 1.5 * math.pi, math.pi]

NORM_UPPER = 1.0
NORM_LOWER = -1.0

ROT_DIM_DICT = {
    'quat': 4,
    'ax': 3,
    'euler': 3,
    'mat': 6,
}

def normalize_trans_torch(hand_t):
    t_min = torch.tensor(T_MIN, dtype=hand_t.dtype, device=hand_t.device)
    t_max = torch.tensor(T_MAX, dtype=hand_t.dtype, device=hand_t.device)
    t = torch.div((hand_t - t_min), (t_max - t_min))
    t = t * (NORM_UPPER - NORM_LOWER) - (NORM_UPPER - NORM_LOWER) / 2
    return t

def denormalize_trans_torch(hand_t):
    t_min = torch.tensor(T_MIN, dtype=hand_t.dtype, device=hand_t.device)
    t_max = torch.tensor(T_MAX, dtype=hand_t.dtype, device=hand_t.device)
    t = hand_t + (NORM_UPPER - NORM_LOWER) / 2
    t /= (NORM_UPPER - NORM_LOWER)
    t = t * (t_max - t_min) + t_min
    return t

def normalize_trans_numpy(hand_t):
    t_min = np.array(T_MIN, dtype=hand_t.dtype)
    t_max = np.array(T_MAX, dtype=hand_t.dtype)
    t = (hand_t - t_min) / (t_max - t_min)
    t = t * (NORM_UPPER - NORM_LOWER) - (NORM_UPPER - NORM_LOWER) / 2
    return t

def denormalize_trans_numpy(hand_t):
    t_min = np.array(T_MIN, dtype=hand_t.dtype)
    t_max = np.array(T_MAX, dtype=hand_t.dtype)
    t = hand_t + (NORM_UPPER - NORM_LOWER) / 2
    t /= (NORM_UPPER - NORM_LOWER)
    t = t * (t_max - t_min) + t_min
    return t

def normalize_param_torch(hand_param):
    hand_pose_min = torch.tensor(HAND_POSE_MIN, dtype=hand_param.dtype, device=hand_param.device)
    hand_pose_max = torch.tensor(HAND_POSE_MAX, dtype=hand_param.dtype, device=hand_param.device)
    p = torch.div((hand_param - hand_pose_min), (hand_pose_max - hand_pose_min))
    p = p * (NORM_UPPER - NORM_LOWER) - (NORM_UPPER - NORM_LOWER) / 2
    return p

def denormalize_param_torch(hand_param):
    hand_pose_min = torch.tensor(HAND_POSE_MIN, dtype=hand_param.dtype, device=hand_param.device)
    hand_pose_max = torch.tensor(HAND_POSE_MAX, dtype=hand_param.dtype, device=hand_param.device)
    p = hand_param + (NORM_UPPER - NORM_LOWER) / 2
    p /= (NORM_UPPER - NORM_LOWER)
    p = p * (hand_pose_max - hand_pose_min) + hand_pose_min
    return p

def normalize_param_numpy(hand_param):
    hand_pose_min = np.array(HAND_POSE_MIN, dtype=hand_param.dtype)
    hand_pose_max = np.array(HAND_POSE_MAX, dtype=hand_param.dtype)
    p = (hand_param - hand_pose_min) / (hand_pose_max - hand_pose_min)
    p = p * (NORM_UPPER - NORM_LOWER) - (NORM_UPPER - NORM_LOWER) / 2
    return p

def denormalize_param_numpy(hand_param):
    hand_pose_min = np.array(HAND_POSE_MIN, dtype=hand_param.dtype)
    hand_pose_max = np.array(HAND_POSE_MAX, dtype=hand_param.dtype)
    p = hand_param + (NORM_UPPER - NORM_LOWER) / 2
    p /= (NORM_UPPER - NORM_LOWER)
    p = p * (hand_pose_max - hand_pose_min) + hand_pose_min
    return p

def normalize_rot_torch(hand_r, rot_type='quat'):
    if rot_type == 'mat':
        return normalize_rot6d_torch(hand_r)
    hand_r_min = torch.tensor(eval(f"R_MIN_{rot_type.upper()}"), dtype=hand_r.dtype, device=hand_r.device)
    hand_r_max = torch.tensor(eval(f"R_MAX_{rot_type.upper()}"), dtype=hand_r.dtype, device=hand_r.device)
    r = torch.div((hand_r - hand_r_min), (hand_r_max - hand_r_min))
    r = r * (NORM_UPPER - NORM_LOWER) - (NORM_UPPER - NORM_LOWER) / 2
    return r

def denormalize_rot_torch(hand_r, rot_type='quat'):
    if rot_type == 'mat':
        return normalize_rot6d_torch(hand_r)
    hand_r_min = torch.tensor(eval(f"R_MIN_{rot_type.upper()}"), dtype=hand_r.dtype, device=hand_r.device)
    hand_r_max = torch.tensor(eval(f"R_MAX_{rot_type.upper()}"), dtype=hand_r.dtype, device=hand_r.device)
    r = hand_r + (NORM_UPPER - NORM_LOWER) / 2
    r /= (NORM_UPPER - NORM_LOWER)
    r = r * (hand_r_max - hand_r_min) + hand_r_min
    return r

def normalize_rot_numpy(hand_r, rot_type='quat'):
    if rot_type == 'mat':
        return normalize_rot6d_numpy(hand_r)
    hand_r_min = np.array(eval(f"R_MIN_{rot_type.upper()}"), dtype=hand_r.dtype)
    hand_r_max = np.array(eval(f"R_MAX_{rot_type.upper()}"), dtype=hand_r.dtype)
    r = (hand_r - hand_r_min) / (hand_r_max - hand_r_min)
    r = r * (NORM_UPPER - NORM_LOWER) - (NORM_UPPER - NORM_LOWER) / 2
    return r

def denormalize_rot_numpy(hand_r, rot_type='quat'):
    if rot_type == 'mat':
        return normalize_rot6d_numpy(hand_r)
    hand_r_min = np.array(eval(f"R_MIN_{rot_type.upper()}"), dtype=hand_r.dtype)
    hand_r_max = np.array(eval(f"R_MAX_{rot_type.upper()}"), dtype=hand_r.dtype)
    r = hand_r + (NORM_UPPER - NORM_LOWER) / 2
    r /= (NORM_UPPER - NORM_LOWER)
    r = r * (hand_r_max - hand_r_min) + hand_r_min
    return r

def decompose_hand_param(hand_pose, rot_type='quat', normalize=True):
    "normalized = True if the params need normalization"
    rot_dim = ROT_DIM_DICT[rot_type.lower()]
    if isinstance(hand_pose, torch.Tensor):
        hand_t, hand_r, hand_param = hand_pose.split((3, rot_dim, 22), dim=-1)
        if normalize:
            hand_t = normalize_trans_torch(hand_t)
            hand_r = normalize_rot_torch(hand_r, rot_type=rot_type)
            hand_param = normalize_param_torch(hand_param)
    elif isinstance(hand_pose, np.ndarray):
        hand_t, hand_r, hand_param = np.split(hand_pose, [3, 3+rot_dim], axis=-1)
        if normalize:
            hand_t = normalize_trans_numpy(hand_t)
            hand_r = normalize_rot_numpy(hand_r, rot_type=rot_type)
            hand_param = normalize_param_numpy(hand_param)
    else:
        raise NotImplementedError

    return hand_t, hand_r, hand_param


def compose_hand_param(hand_t, hand_r, hand_param, rot_type='quat', normalized=False):
    "normalized = True if the params are already normalized"
    if isinstance(hand_t, torch.Tensor):
        if not normalized:
            hand_t = denormalize_trans_torch(hand_t)
            hand_r = denormalize_rot_torch(hand_r, rot_type=rot_type)
            hand_param = denormalize_param_torch(hand_param)
        hand = torch.cat([hand_t, hand_r, hand_param], dim=-1)
    elif isinstance(hand_t, np.ndarray):
        if not normalized:
            hand_t = denormalize_trans_numpy(hand_t)
            hand_r = denormalize_rot_numpy(hand_r, rot_type=rot_type)
            hand_param = denormalize_param_numpy(hand_param)
        hand = np.concatenate([hand_t, hand_r, hand_param], axis=-1)
    else:
        raise NotImplementedError
    
    return hand



