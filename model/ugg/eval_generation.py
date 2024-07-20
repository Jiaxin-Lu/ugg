import os
import sys
from os.path import join as pjoin

base_dir = os.path.dirname(__file__)
sys.path.append(pjoin(base_dir, '..'))
sys.path.append(pjoin(base_dir, '..', '..'))
sys.path.append(pjoin(base_dir, '../../LION/'))

import pickle
import time
from datetime import datetime

import numpy as np
import pytorch_lightning as pl
import torch
from tqdm import tqdm

from model.ugg.discriminator import DiscriminatorTrainer
from model.ugg.generation_contact_tester import UGGGenerationTester
from model.ugg.optimizer.post_optimization import (post_optimization_joint,
                                                   post_optimization_pc,
                                                   post_optimization_pose)
from utils.hand_helper import compose_hand_param, decompose_hand_param
from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.utils import (dict_to_cpu, dict_to_device, dict_to_numpy,
                         extract_batch0, load_model, save_pcd)

NOW_TIME = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


torch.set_float32_matmul_precision('high')


def joint_test(cfg):
    print("Test joint")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device", device)

    model_save_path = cfg.MODEL_SAVE_PATH
    ouptut_path = cfg.OUTPUT_PATH
    ckp_path = load_model(None, cfg.WEIGHT_FILE, model_save_path)
    print(f"ckp_path: {ckp_path}")
    model = UGGGenerationTester(cfg, ckpt=ckp_path, device=device)
    model = model.to(device)
    model.eval()
    
    hand_model = HandModel(
        mjcf_path='hand_model_mjcf/shadow_hand_wrist_free.xml',
        mesh_path='hand_model_mjcf/meshes',
        contact_points_path='hand_model_mjcf/contact_points.json',
        penetration_points_path='hand_model_mjcf/penetration_points.json',
        n_surface_points=1024,
        device=device
    )
    
    print("finish setting -----")
    
    gen_dict = dict()
    B = cfg.GEN_OBJECT_NUM
    assert B <= 1000

    test_cfg = cfg.MODEL.test

    file_suffix = ""
    if cfg.LOG_FILE_NAME is not None and len(cfg.LOG_FILE_NAME) > 0:
        file_suffix += f"_{cfg.LOG_FILE_NAME}"
    file_suffix += f"_{NOW_TIME}"
    os.makedirs(os.path.join(ouptut_path, 'joint'), exist_ok=True)
    if cfg.FOLDER is not None:
        os.makedirs(os.path.join(ouptut_path, 'joint', cfg.FOLDER), exist_ok=True)
        save_file = os.path.join(ouptut_path, 'joint', cfg.FOLDER, f"joint{file_suffix}.pk")
    else:
        save_file = os.path.join(ouptut_path, 'joint', f"joint{file_suffix}.pk")

    seed_dict = model.get_generation_dict(B)
    seed_dict = dict_to_device(seed_dict, device)
    data_dict = seed_dict
    
    ret_dict = model.test_step(data_dict, task=['joint'])
    
    pc_actual = ret_dict['joint']['pc_actual'].detach().clone()
    contact_map_actual = ret_dict['joint']['contact_map_actual'].detach().clone() if model.gen_contact else None

    hand_param = ret_dict['joint']['hand_param_composed'].detach().clone()
    hand_model.set_parameters(hand_param)

    hand_opt, pc_opt, loss_list, hand_pose_list, pc_list = post_optimization_joint(test_cfg, hand_param, hand_model, pc_actual, contact_points=contact_map_actual, steps=test_cfg.opt_iter)
    
    gen_dict = {
        'pc_out': ret_dict['joint']['pc_final'],
        'pc_opt': pc_opt,
        'hand_param_out': ret_dict['joint']['hand_param_composed'],
        'hand_param_opt': hand_opt,
        'contact_map': ret_dict['joint']['contact_map_actual'] if model.gen_contact else None,
        'seed_dict': seed_dict,
    }
    gen_dict = dict_to_numpy(gen_dict)

    with open(save_file, 'wb') as f:
        pickle.dump(gen_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("Done test.")


def hand2obj_test(cfg):
    print("Test hand2obj")
    train_loader, val_loader = build_dataloader(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device", device)

    model_save_path = cfg.MODEL_SAVE_PATH
    ouptut_path = cfg.OUTPUT_PATH
    ckp_path = load_model(None, cfg.WEIGHT_FILE, model_save_path)
    print(f"ckp_path: {ckp_path}")
    model = UGGGenerationTester(cfg, ckpt=ckp_path, device=device)
    model = model.to(device)
    model.eval()
    
    hand_model = HandModel(
        mjcf_path='hand_model_mjcf/shadow_hand_wrist_free.xml',
        mesh_path='hand_model_mjcf/meshes',
        contact_points_path='hand_model_mjcf/contact_points.json',
        penetration_points_path='hand_model_mjcf/penetration_points.json',
        n_surface_points=1024,
        device=device
    )

    print("finish setting -----")
    
    assert cfg.BATCH_SIZE == 1
    gen_dict = dict()
    B = cfg.GEN_OBJECT_NUM

    test_cfg = cfg.MODEL.test
    
    file_suffix = ""
    if cfg.LOG_FILE_NAME is not None and len(cfg.LOG_FILE_NAME) > 0:
        file_suffix += f"_{cfg.LOG_FILE_NAME}"
    file_suffix += f"_{NOW_TIME}"
    os.makedirs(os.path.join(ouptut_path, 'test_hand2obj'), exist_ok=True)
    if cfg.FOLDER is not None:
        os.makedirs(os.path.join(ouptut_path, 'test_hand2obj', cfg.FOLDER), exist_ok=True)
        save_file = os.path.join(ouptut_path, 'test_hand2obj', cfg.FOLDER, f"test_hand2obj{file_suffix}.pk")
    else:
        save_file = os.path.join(ouptut_path, 'test_hand2obj', f"test_hand2obj{file_suffix}.pk")

    for batch_idx, (data_dict) in tqdm(enumerate(val_loader), desc=f"Hand2Ojbect on Validation", total=len(val_loader)):
        data_dict = extract_batch0(data_dict)
        mesh_code = data_dict['grasp_code'].split("#")[0]
        if mesh_code not in gen_dict:
            data_dict = dict_to_device(data_dict, device)
            seed_dict = model.get_generation_dict(B)
            seed_dict = dict_to_device(seed_dict, device)
            data_dict.update(seed_dict)
            
            ret_dict = model.test_step(data_dict, task=['hand2obj'])
            
            hand_param_in = data_dict['ori_hand_pose']
            pc_actual = ret_dict['hand2obj']['pc_actual'].detach().clone()
            contact_map_actual = ret_dict['hand2obj']['contact_map_actual'].detach().clone() if model.gen_contact else None
            hand_model.set_parameters(hand_param_in)
            
            pc_opt, hand_opt, loss_list, pc_opt_list, hand_opt_list = post_optimization_pc(test_cfg, pc_actual, hand_param_in, hand_model, contact_map_actual, steps=test_cfg.opt_iter)
            
            current_gen_dict = {
                'pc_out': ret_dict['hand2obj']['pc_final'],
                'pc_opt': pc_opt,
                'hand_param_in': hand_param_in,
                'hand_param_out': ret_dict['hand2obj']['hand_param_composed'],
                'hand_param_opt': hand_opt,
                'contact_map': ret_dict['hand2obj']['contact_map_actual'] if model.gen_contact else None,
                'seed_dict': seed_dict,
            }

            gen_dict[mesh_code] = dict_to_numpy(current_gen_dict)
        else:
            print("duplicated test")
        if (batch_idx + 1) % 25 == 0:
            with open(save_file, 'wb') as f:
                pickle.dump(gen_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(save_file, 'wb') as f:
        pickle.dump(gen_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("Done test.")


def obj2hand_test(cfg, task='obj2hand'):
    print(f"Test {task}")
    train_loader, val_loader = build_dataloader(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device", device)

    model_save_path = cfg.MODEL_SAVE_PATH
    ouptut_path = cfg.OUTPUT_PATH
    ckp_path = load_model(None, cfg.WEIGHT_FILE, model_save_path)
    print(f"ckp_path: {ckp_path}")
    model = UGGGenerationTester(cfg, ckpt=ckp_path, device=device)
    model = model.to(device)
    model.eval()
    
    if len(cfg.MODEL.diffusion.disc.discriminator_checkpoint):
        print(f"Use discriminator: {cfg.MODEL.diffusion.disc.discriminator_checkpoint}")
        disc = DiscriminatorTrainer.load_from_checkpoint(cfg.MODEL.diffusion.disc.discriminator_checkpoint, cfg=cfg)
        discriminator = disc.discriminator.to(device)

        discriminator.eval()
        thr = 0.5
    else:
        discriminator = None
        
    hand_model = HandModel(
        mjcf_path='hand_model_mjcf/shadow_hand_wrist_free.xml',
        mesh_path='hand_model_mjcf/meshes',
        contact_points_path='hand_model_mjcf/contact_points.json',
        penetration_points_path='hand_model_mjcf/penetration_points.json',
        n_surface_points=1024,
        device=device
    )
    
    object_model = ObjectModel(data_dir=cfg.DATA.DATA_DIR,
                               mesh_dir=cfg.DATA.MESH_DIR,
                               pc_num_points=2048, 
                               device=device,
                               dataset_name=cfg.DATASET.lower())


    print("finish setting -----")
    
    assert cfg.BATCH_SIZE == 1
    
    gen_dict = dict()
    
    B = val_loader.dataset.gen_grasp_num

    test_cfg = cfg.MODEL.test

    file_suffix = ""
    if cfg.LOG_FILE_NAME is not None and len(cfg.LOG_FILE_NAME) > 0:
        file_suffix += f"_{cfg.LOG_FILE_NAME}"
    file_suffix += f"_{NOW_TIME}"
    os.makedirs(os.path.join(ouptut_path, 'test_obj2hand'), exist_ok=True)

    if cfg.FOLDER is not None:
        os.makedirs(os.path.join(ouptut_path, 'test_obj2hand', cfg.FOLDER), exist_ok=True)
        save_file = os.path.join(ouptut_path, 'test_obj2hand', cfg.FOLDER, f"test_obj2hand{file_suffix}.pk")
    else:
        save_file = os.path.join(ouptut_path, 'test_obj2hand', f"test_obj2hand{file_suffix}.pk")

    print(f"use task {cfg.MODEL.diffusion.task}")

    for batch_idx, (data_dict) in tqdm(enumerate(val_loader), desc=f"Obj2Hand on Validation", total=len(val_loader)):
        data_dict = extract_batch0(data_dict)
        data_dict = dict_to_device(data_dict, device)
        seed_dict = model.get_generation_dict(B)
        seed_dict = dict_to_device(seed_dict, device)
        seed_dict.pop('pc_global_latent_gen')
        seed_dict.pop('pc_local_latent_gen')
        
        data_dict.update(seed_dict)

        mesh_code = data_dict['grasp_code'].split("#")[0]
        
        ret_dict = model.test_step(data_dict, task=[task])
        
        hand_param = ret_dict[task]['hand_param_composed'].detach().clone()
        contact_map = ret_dict[task]['contact_map_actual'].detach().clone() if model.gen_contact else None
        
        hand_model.set_parameters(hand_param)
        object_model.set_parameters(mesh_code + "#-1", object_sc=data_dict['object_sc'])
        
        hand_opt, loss_list, hand_pose_list = post_optimization_pose(test_cfg, hand_param, hand_model, object_model, contact_points=contact_map, steps=test_cfg.opt_iter)
        
        
        hand_param = ret_dict[task]['hand_param_composed'].detach().clone()
        current_gen_dict = {
            'hand_param_out': hand_param,
            'hand_param_opt': hand_opt,
            'object_pc': data_dict['object_pc'],
            'object_sc': data_dict['object_sc'],
            'contact_map': ret_dict[task]['contact_map_actual'] if model.gen_contact else None,
            'seed_dict': seed_dict,
            # 'hand_param_opt_list': hand_pose_list,  # if you want to save the intermediate results, uncomment this line
        }
        
        # For discriminator, should use normalized pc (6.6x) and decomposed hand param 
        if discriminator is not None:
            hand_t, hand_R, hand_param = decompose_hand_param(hand_param, rot_type='mat')
            contact_map = ret_dict[task]['contact_map_actual'].detach().clone() if model.gen_contact else None
            object_pc = data_dict['object_pc'].clone()
            disc_test_dict = {
                'hand_t': hand_t, 
                'hand_R': hand_R,
                'hand_param': hand_param,
                'object_pc': object_pc,
                'contact_map': contact_map if model.gen_contact else None,
            }
            with torch.no_grad():
                disc_out_dict = discriminator(disc_test_dict)
                pred = disc_out_dict['pred']
                pred = torch.sigmoid(pred)
            current_gen_dict['disc_pred'] = pred
        
        gen_dict[mesh_code] = dict_to_numpy(current_gen_dict)

        if (batch_idx + 1) % 20 == 0:
            with open(save_file, 'wb') as f:
                pickle.dump(gen_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(save_file, 'wb') as f:
        pickle.dump(gen_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("Done test.")
    

if __name__ == '__main__':
    from dataset import build_dataloader
    from utils.config import cfg
    from utils.dup_stdout_manager import DupStdoutFileManager
    from utils.parse_args import parse_args
    from utils.print_easydict import print_easydict

    args = parse_args("Diffusion")
    cfg.FOLDER = args.folder

    pl.seed_everything(cfg.RANDOM_SEED)

    file_suffix = ""
    if cfg.LOG_FILE_NAME is not None and len(cfg.LOG_FILE_NAME) > 0:
        file_suffix += "_{}".format(cfg.LOG_FILE_NAME)
    file_suffix += f"_{NOW_TIME}"
    full_log_name = f"eval_log{file_suffix}"

    os.makedirs(os.path.join(cfg.OUTPUT_PATH, "eval_log"), exist_ok=True)
    
    with DupStdoutFileManager(os.path.join(cfg.OUTPUT_PATH, "eval_log", f"{full_log_name}.log")) as _:

        print_easydict(cfg)
        
        task = cfg.MODEL.diffusion.task
        print("Tasks: ", task)
        for t in task:
            if 'obj2hand' in t:
                obj2hand_test(cfg, t)
            if 'hand2obj' in t:
                hand2obj_test(cfg)
            if 'joint' in t:
                joint_test(cfg)
    
