import os.path
import pickle
import random

import numpy as np
import torch
import transforms3d
import trimesh
from loguru import logger
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader, Dataset

from utils.hand_helper import decompose_hand_param

translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
joint_names = [
    'robot0:WRJ1', 'robot0:WRJ0',
    'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
    'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
    'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
    'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
    'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
]

class DexGraspDataset(Dataset):
    def __init__(self,
                 cfg,
                 mode,

                hand_ratio=-1,
                object_ratio=-1,
                *args, **kwargs
                ):
        super().__init__()
        self.mode = mode
        self.data_dir = cfg.DATA.DATA_DIR
        self.mesh_dir = cfg.DATA.MESH_DIR
        self.splits_dir = cfg.DATA.SPLITS_DIR
        self.pc_num_points = cfg.DATA.PC_NUM_POINTS
        dataset_length = eval(f'cfg.DATA.{mode.upper()}_LENGTH')
        
        self.rot_type = cfg.DATA.ROT_TYPE
        logger.info("dataset.rot_type: {}", self.rot_type)

        hand_ratio=cfg.DATA.HAND_RATIO
        object_ratio=cfg.DATA.OBJECT_RATIO

        self.use_point_cloud=cfg.DATA.USE_POINT_CLOUD
        self.use_precompute_pc=cfg.DATA.USE_PRECOMPUTE_PC
        self.normalize_pc=cfg.DATA.NORMALIZE_PC
        self.apply_random_rot=cfg.DATA.APPLY_RANDOM_ROT


        logger.info(f'apply random rot {self.apply_random_rot}')
        if self.apply_random_rot:
            assert self.use_precompute_pc is False

        self.grasp_code_idx_list, self.mesh_dir_list = self.read_data(self.data_dir, self.mesh_dir, self.splits_dir, mode, object_ratio=object_ratio, hand_ratio=hand_ratio)
        self.dataset_length = len(self.grasp_code_idx_list)
        logger.info('initial dataset_lenght: {}', self.dataset_length)

        if 0 < dataset_length < self.dataset_length:
            rand_idx = random.sample(list(range(self.dataset_length)), dataset_length)
            rand_idx.sort()
            self.grasp_code_idx_list = [self.grasp_code_idx_list[i] for i in rand_idx]
            self.mesh_dir_list = [self.mesh_dir_list[i] for i in rand_idx]
            self.dataset_length = len(self.grasp_code_idx_list)
            logger.info(f'update dataset length to {self.dataset_length}')
        
        self.pre_saved_grasp_data = []
        self.pre_saved_pc = {}

    def read_data(self, data_dir, mesh_dir, splits_dir, mode, object_ratio=-1, hand_ratio=-1):
        pk_file_name = f'saved_dexgraspnet_data_{mode}'
        if object_ratio > 0:
            pk_file_name = pk_file_name + f"_o{object_ratio}"
        if hand_ratio > 0:
            pk_file_name = pk_file_name + f"_h{hand_ratio}"
        pk_file_name += '.pk'
        if os.path.exists(os.path.join(splits_dir, pk_file_name)):
            print('load from existing presaved file')
            with open(os.path.join(splits_dir, pk_file_name), 'rb') as f:
                data_dict = pickle.load(f)
            return data_dict['grasp_code_idx_list'], data_dict['mesh_dir_list']
        
        print("generate presaved files")
        grasp_code_list = []
        with open(os.path.join(splits_dir, f'split_dexgraspnet_{mode}.txt'), 'r') as f:
            grasp_code_list += f.read().splitlines()
        grasp_code_list = [grasp_code for grasp_code in grasp_code_list if len(grasp_code) > 1]
        grasp_code_list.sort()
        
        grasp_code_idx_list = []
        mesh_dir_list = []
        for grasp_code in grasp_code_list:
            grasp_data = np.load(os.path.join(data_dir, grasp_code + '.npy'), allow_pickle=True)
            len_grasp = len(grasp_data)
            if 0 < hand_ratio < 1:
                id_list = random.sample(list(range(len_grasp)), int(len_grasp * hand_ratio))
            elif hand_ratio >= 1:
                id_list = list(range(hand_ratio))
                # id_list = random.sample(list(range(len_grasp)), int(hand_ratio))
            else:
                id_list = list(range(len_grasp))
            grasp_code_idx_list.extend([(grasp_code, idx) for idx in id_list])
            mesh_obj_path = os.path.join(mesh_dir, grasp_code, "coacd/decomposed.obj")
            mesh_dir_list.extend([mesh_obj_path for idx in range(len_grasp)])

        data_dict = {
            'grasp_code_idx_list': grasp_code_idx_list,
            'mesh_dir_list': mesh_dir_list
        }
        print("finish, saving...")
        with open(os.path.join(splits_dir, pk_file_name), 'wb') as f:
            pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        return grasp_code_idx_list, mesh_dir_list

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        if len(self.pre_saved_grasp_data):
            grasp_code, grasp_idx = self.grasp_code_idx_list[idx]
            grasp_data = self.pre_saved_grasp_data[idx]
            qpos = grasp_data['qpos']
        else:
            grasp_code, grasp_idx = self.grasp_code_idx_list[idx]
            grasp_data = np.load(os.path.join(self.data_dir, grasp_code + '.npy'), allow_pickle=True)
            grasp_data = grasp_data[grasp_idx]
            qpos = grasp_data['qpos']
        translation = np.array([qpos[name] for name in translation_names], dtype=np.float32)
        r_rand = None
        if self.apply_random_rot:
            r_rand = Rotation.random().as_matrix()
            r_rand = r_rand.astype(np.float32)
            translation = translation @ r_rand
        # print(self.rot_type)
        if self.rot_type == 'ax':
            rotation = transforms3d.euler.euler2mat(*[qpos[name] for name in rot_names])
            if r_rand is not None:
                rotation = r_rand.T @ rotation
            vector, theta = transforms3d.axangles.mat2axangle(rotation)
            rotation = (np.array(vector) / np.linalg.norm(vector) * theta)
        elif self.rot_type == 'quat':
            rotation = transforms3d.euler.euler2quat(*[qpos[name] for name in rot_names])
            if r_rand is not None:
                rotation = r_rand.T @ rotation
            rotation = np.array(rotation)
            rotation = rotation / np.linalg.norm(rotation)
            # print('rotation_norm', np.linalg.norm(rotation))
        elif self.rot_type == 'euler':
            rotation = np.array([qpos[name] for name in rot_names])            
            if r_rand is not None:
                assert False
        elif self.rot_type == 'mat':
            rotation = transforms3d.euler.euler2mat(*[qpos[name] for name in rot_names])
            if r_rand is not None:
                rotation = r_rand.T @ rotation
            rotation = rotation[:, :2].T.ravel().tolist()
            rotation = np.array(rotation)

        joint_angles = np.array([qpos[name] for name in joint_names[2:]], np.float32)
        hand_pose = torch.tensor(translation.tolist() + rotation.tolist() + joint_angles.tolist(), dtype=torch.float)

        rand_id = random.randint(0, 9)
        if self.use_precompute_pc and self.pre_saved_pc:
            latent_dict = self.pre_saved_pc[grasp_code]
            latent_list = latent_dict[grasp_data['scale']]
            latent_list = [[tt[rand_id] for tt in t] for t in latent_list]
        elif self.use_precompute_pc and self.pc_num_points == 2048:
            if self.use_point_cloud:
                pc_path = os.path.join(self.mesh_dir, grasp_code, f"coacd/pc_2048_{rand_id:03d}.npy")
                samples = np.load(pc_path)
            # loaded point cloud is of original shape
            # assert samples.shape[0] == self.pc_num_points
            if self.normalize_pc:
                with open(os.path.join(self.mesh_dir, grasp_code, f"coacd/pc_norm_latent_LION_{rand_id:03d}.pk"), 'rb') as f:
                    latent_dict = pickle.load(f)
                latent_list = latent_dict[grasp_data['scale']]
            else:
                with open(os.path.join(self.mesh_dir, grasp_code, f"coacd/pc_latent_LION_{rand_id:03d}.pk"), 'rb') as f:
                    latent_dict = pickle.load(f)
                latent_list = latent_dict[grasp_data['scale']]
        else:
            mesh_obj_path = self.mesh_dir_list[idx]
            mesh = trimesh.load(mesh_obj_path, force='mesh')
            samples, fid = mesh.sample(self.pc_num_points, return_index=True)
            latent_list = None

        
        ori_hand_pose = hand_pose
        hand_t, hand_R, hand_param = decompose_hand_param(hand_pose, rot_type=self.rot_type, normalize=True)
        hand_pose = torch.cat([hand_t, hand_R, hand_param], dim=-1)

        data_dict = {
            'ori_hand_pose': ori_hand_pose,
            'hand_pose': hand_pose,
            'grasp_code': f"{grasp_code}#{grasp_idx}",
        }

        if latent_list is not None:
            data_dict.update({
                'z_mu_global': latent_list[0][1],
                'z_sigma_global': latent_list[0][2],
                'z_mu_local': latent_list[1][1],
                'z_sigma_local': latent_list[1][2],
            })
            
        if r_rand is not None:
            data_dict.update({
                'r_rand': r_rand,
            })
            
        if self.use_point_cloud:
            mesh_obj_path = self.mesh_dir_list[idx]
            mesh = trimesh.load(mesh_obj_path, force='mesh')
            samples, fid = mesh.sample(self.pc_num_points, return_index=True)
            
            object_pc = torch.tensor(samples, dtype=torch.float) * grasp_data['scale']
            if self.normalize_pc:
                object_pc *= 6.6
            data_dict['object_pc'] = object_pc
            data_dict['object_sc'] = grasp_data['scale']

        return data_dict


def build_dexgraspnet_dataloader(cfg):
    data_dict = dict(
        cfg=cfg,
        mode="train",
    )
    train_set = DexGraspDataset(**data_dict)
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
    test_set = DexGraspDataset(**data_dict)
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.NUM_WORKERS > 0)
    )
    return train_loader, test_loader
