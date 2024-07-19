import os
import pickle
import random

import numpy as np
import torch
import transforms3d
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from utils.hand_helper import decompose_hand_param, hand_translation_names, hand_rot_names, hand_joint_names

class DiscriminatorDataset(Dataset):
    def __init__(self, cfg, mode, ori_data_dir='data/discriminator_data', *args, **kwargs):
        super().__init__()
        self.mode = mode
        self.file_dir = cfg.DATA.FILE_DIR
        self.mesh_dir = cfg.DATA.MESH_DIR
        self.normalize_pc = cfg.DATA.NORMALIZE_PC
        self.decoded_hand = cfg.DATA.DECODED_HAND
        self.rot_type = 'mat'
        with open(os.path.join(self.file_dir, f"disc_dict_{mode}.pk"), 'rb') as f:
            self.data_dict = pickle.load(f)
        if self.decoded_hand:
            logger.info("add the original data into discriminator data")
            self.update_with_original_data(ori_data_dir)
        
        self.code_list = code_list = list(self.data_dict.keys())
        self.code_pair_list = [(code, idx) for code in code_list for idx in range(self.data_dict[code]['valid'].shape[0])]
        self.dataset_length = len(self.code_pair_list)
        logger.info(f"dataset_length {self.dataset_length}")
        
    def update_with_original_data(self, ori_data_dir):
        ori_data_path = os.path.join(ori_data_dir, f"discriminator_data_on_original_dataset_{self.mode}.pk")
        if os.path.exists(ori_data_path):
            logger.info("load original data")
            with open(ori_data_path, 'rb') as f:
                combined_dict = pickle.load(f)
        else:
            combined_dict = construct_original_discriminator_data(self.mode)
            with open(ori_data_path, 'wb') as f:
                pickle.dump(combined_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
                
        for code in combined_dict.keys():
            if code not in self.data_dict.keys():
                self.data_dict[code] = combined_dict[code]
            else:
                rel_dict = self.data_dict[code]
                c_dict = combined_dict[code]
                hand_param = rel_dict['hand_param']
                valid = rel_dict['valid']
                valid_refine = rel_dict['valid_refine']
                object_sc = rel_dict['object_sc']
                rel_dict['hand_param'] = np.concatenate((hand_param, c_dict['hand_param']), axis=0)
                rel_dict['valid'] = np.concatenate((valid, c_dict['valid']), axis=0)
                rel_dict['valid_refine'] = np.concatenate((valid_refine, c_dict['valid']), axis=0)
                rel_dict['object_sc'] = np.concatenate((object_sc, c_dict['object_sc'].reshape(-1, 1)), axis=0)
                self.data_dict[code] = rel_dict
    
    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        code, index = self.code_pair_list[idx]
        data = self.data_dict[code]

        object_sc = data['object_sc'][index].item()
        rand_id = random.randint(0, 9)
        pc_path = os.path.join(self.mesh_dir, code, f"coacd/pc_2048_{rand_id:03d}.npy")
        samples = np.load(pc_path) * object_sc
        if self.normalize_pc:
            with open(os.path.join(self.mesh_dir, code, f"coacd/pc_norm_latent_LION_{rand_id:03d}.pk"), 'rb') as f:
                latent_dict = pickle.load(f)
            dict_keys = list(latent_dict.keys())
            sc = dict_keys[np.argmin(np.abs(np.array(dict_keys)-object_sc))]
            latent_list = latent_dict[sc]
            samples = samples * 6.6
        else:
            with open(os.path.join(self.mesh_dir, code, f"coacd/pc_latent_LION_{rand_id:03d}.pk"), 'rb') as f:
                latent_dict = pickle.load(f)
            dict_keys = list(latent_dict.keys())
            sc = dict_keys[np.argmin(np.abs(np.array(dict_keys)-object_sc))]
            latent_list = latent_dict[sc]

        valid = data['valid_refine'][index]
        valid = valid.astype(np.float32)
        data_dict = {
            'valid': valid,
            'z_mu_global': latent_list[0][1],
            'z_sigma_global': latent_list[0][2],
            'z_mu_local': latent_list[1][1],
            'z_sigma_local': latent_list[1][2],
            'object_sc': np.array(object_sc),
            'object_pc': samples.astype(np.float32),
        }
        if self.decoded_hand:
            hand_param_final = data['hand_param'][index].reshape(1, -1)
            hand_t, hand_r, hand_param = decompose_hand_param(hand_param_final, rot_type='mat')
            data_dict.update({
                'hand_param': hand_param[0],
                'hand_R': hand_r[0],
                'hand_t': hand_t[0],
            })
        else:
            data_dict.update({
                'hand_param': data['hand_param_seed'][index],
                'hand_R': data['hand_R_seed'][index],
                'hand_t': data['hand_t_seed'][index],
            })

        return data_dict
        
        
def build_discriminator_dataset(cfg):
    data_dict = dict(
        cfg=cfg,
        mode='train',
    )
    train_set = DiscriminatorDataset(**data_dict)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )
    
    data_dict['mode'] = "test"
    test_set = DiscriminatorDataset(**data_dict)
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )
    return train_loader, test_loader


def construct_original_discriminator_data(mode, rot_type='mat'):
    assert rot_type == 'mat', "the model only supports mat6d for rotation now"
    from .dataset_config import dataset_cfg
    splits_dir = dataset_cfg.DEXGRASPNET.SPLITS_DIR
    data_dir = dataset_cfg.DEXGRASPNET.DATA_DIR
    grasp_code_list = []
    with open(os.path.join(splits_dir, f'split_dexgraspnet_{mode}.txt'), 'r') as f:
        grasp_code_list += f.read().splitlines()
    grasp_code_list = [grasp_code for grasp_code in grasp_code_list if len(grasp_code) > 1]
    grasp_code_list.sort()

    logger.info(f"try construct {len(grasp_code_list)} grasp codes into discriminator data")
    discriminator_dict = {}
    for grasp_code in grasp_code_list:
        grasp_data = np.load(os.path.join(data_dir, grasp_code + '.npy'), allow_pickle=True)
        len_grasp = len(grasp_data)
        id_list = list(range(len_grasp))
        hand_pose_list = []
        object_sc_list = []
        for id in id_list:
            grasp_data_id = grasp_data[id]
            qpos = grasp_data_id['qpos']
            translation = np.array([qpos[name] for name in hand_translation_names], dtype=np.float32)
            rotation = transforms3d.euler.euler2mat(*[qpos[name] for name in hand_rot_names])
            rotation = rotation[:, :2].T.ravel().tolist()
            rotation = np.array(rotation)
            joint_angles = np.array([qpos[name] for name in hand_joint_names[2:]], np.float32)
            hand_pose_list.append(translation.tolist() + rotation.tolist() + joint_angles.tolist())
            object_sc_list.append(grasp_data_id['scale'])
        hand_pose = np.array(hand_pose_list, dtype=np.float32)
        object_sc = np.array(object_sc_list, dtype=np.float32)
        valid = np.ones(len_grasp, dtype=np.float32)
        discriminator_dict[grasp_code] = {
            'hand_param': hand_pose,
            'object_sc': object_sc,
            'valid': valid,
        }

    assert len(discriminator_dict.keys()) == len(grasp_code_list), "the number of grasp codes should be the same as the dataset"
    return discriminator_dict
    
    