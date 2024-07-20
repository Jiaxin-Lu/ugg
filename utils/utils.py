import platform
import os
import torch
import numpy as np


def try_to_device(d, device):
    try:
        d = d.to(device)
    except AttributeError:
        pass
    return d

def try_to_cuda(d):
    try:
        d = d.cuda()
    except AttributeError:
        pass
    return d

def try_to_detach(d):
    try:
        d = d.detach()
    except AttributeError:
        pass
    return d

def try_to_cpu(d):
    try:
        d = d.detach().cpu()
    except AttributeError:
        pass
    return d

def try_to_numpy(d):
    try:
        d = d.detach().cpu().numpy()
    except AttributeError:
        pass
    return d

def try_to_torch(d):
    try:
        d = torch.from_numpy(d)
    except AttributeError:
        pass
    return d
        
def dict_to_device(data_dict, device):
    ret_dict = dict()
    for k, v in data_dict.items():
        if isinstance(v, list):
            v_device = [try_to_device(t, device) for t in v]
            ret_dict[k] = v_device
        elif isinstance(v, dict):
            v_device = dict_to_device(v, device)
            ret_dict[k] = v_device
        else:
            v_device = try_to_device(v, device)
            ret_dict[k] = v_device
    return ret_dict

def dict_to_cuda(data_dict):
    ret_dict = dict()
    for k, v in data_dict.items():
        if isinstance(v, list):
            v_device = [try_to_cuda(t) for t in v]
            ret_dict[k] = v_device
        if isinstance(v, dict):
            v_device = dict_to_cuda(v)
            ret_dict[k] = v_device
        else:
            v_device = try_to_cuda(v)
            ret_dict[k] = v_device
    return ret_dict

def dict_to_detach(data_dict):
    ret_dict = dict()
    for k, v in data_dict.items():
        if isinstance(v, list):
            v_device = [try_to_detach(t) for t in v]
            ret_dict[k] = v_device
        elif isinstance(v, dict):
            v_device = dict_to_detach(v)
            ret_dict[k] = v_device
        else:
            v_device = try_to_detach(v)
            ret_dict[k] = v_device
    return ret_dict

def dict_to_cpu(data_dict):
    ret_dict = dict()
    for k, v in data_dict.items():
        if isinstance(v, list):
            v_device = [try_to_cpu(t) for t in v]
            ret_dict[k] = v_device
        elif isinstance(v, dict):
            v_device = dict_to_cpu(v)
            ret_dict[k] = v_device
        else:
            v_device = try_to_cpu(v)
            ret_dict[k] = v_device
    return ret_dict

def dict_to_numpy(data_dict):
    ret_dict = dict()
    for k, v in data_dict.items():
        if isinstance(v, list):
            v_device = [try_to_numpy(t) for t in v]
            ret_dict[k] = v_device
        elif isinstance(v, dict):
            v_device = dict_to_numpy(v)
            ret_dict[k] = v_device
        else:
            v_device = try_to_numpy(v)
            ret_dict[k] = v_device
    return ret_dict

def extract_batch_num(data_dict, num):
    ret_dict = dict()
    for k, v in data_dict.items():
        if isinstance(v, str):
            ret_dict[k] = v
            continue
        try:
            ret_dict[k] = v[:num]
        except:
            ret_dict[k] = v
    return ret_dict

def extract_batch0(data_dict):
    ret_dict = dict()
    for k, v in data_dict.items():
        try:
            ret_dict[k] = v[0]
        except:
            ret_dict[k] = v
    return ret_dict
    
def square_distance(src, dst):
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, dim=-1)[:, :, None]
    dist += torch.sum(dst ** 2, dim=-1)[:, None, :]

    dist = torch.clamp(dist, min=1e-16, max=None)
    return dist

def cp_some(src, tgt):
    if platform.system().lower() == 'windows':
        cmd = 'copy {} {}'.format(src, tgt)
    else:
        cmd = 'cp {} {}'.format(src, tgt)
    print(cmd)
    os.system(cmd)
    
def load_model(model, weight_file='', model_save_path=''):
    ckp_path = None
    if len(weight_file):  # if specify a weight file, load it
        # check if it has training states, or just a model weight
        ckp = torch.load(weight_file, map_location="cpu")
        # if it has, then it's a checkpoint compatible with pl
        if "state_dict" in ckp.keys():
            ckp_path = weight_file
        # if it's just a weight, then manually load it to the model
        else:
            model.load_state_dict(ckp)
    else:
        ckp_files = os.listdir(model_save_path)
        ckp_files = [
            ckp for ckp in ckp_files if ("model_" in ckp) or ("last" in ckp)
        ]
        if ckp_files:
            ckp_files = sorted(
                ckp_files,
                key=lambda x: os.path.getmtime(os.path.join(model_save_path, x)),
            )
            last_ckp = ckp_files[-1]
            print(f"INFO: automatically detect checkpoint {last_ckp}")
            ckp_path = os.path.join(model_save_path, last_ckp)
    return ckp_path
        
        