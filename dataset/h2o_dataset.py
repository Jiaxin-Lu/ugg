import os.path
import pickle
import random
from loguru import logger
import loguru

import torch
import numpy as np
import transforms3d
from torch.utils.data import Dataset, DataLoader
from utils.hand_helper import decompose_hand_param
import trimesh
from utils.hand_model import HandModel

translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
joint_names = [
    'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
    'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
    'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
    'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
    'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
]
# thin object 
hand_poses_0 = [0.09, 0, 0,
                2., 2., 0.,  
                0.0803,     1.12,       0.33,       0.503, 
                0.0489,     1.09,       0.518,      0.33, 
                0.00349,    1.04,       0.644,      0.322, 
                0.0707,     -0.0454,    0.93,       0.424,      0.958, 
                0.44,       1.22,       0.209,      -0.112,     -0.151,  
                ]

# small object 
hand_poses_1 = [0.07, 0.04, 0.02,
                2., 2., 0.,  
                0.0803,  1.12,  0.361,   0.377, 
                0.0489,  1.09,  0.456,   0.157,
                0.00349,  1.04,  0.526,   0.589, 
                0.0707,  -0.0489,  1.02,   0.518,   0.817, 
                0.0628,  1.22,  0.209,   0.0977,   -0.655,  
               ]

# middle object
hand_poses_2 = [0.07, 0, 0.02,
                2., 2., 0.,  
                0.234,  0.609,  0.754,   0.778, 
                -0.0209,  0.508,  0.895,   0.864,
                0.00349,  1.04,  0.526,   0.589, 
                0.0707,  -0.0489,  1.02,   0.518,   0.817, 
                -0.178,  0.959,  0.0419,   -0.0279,   -0.609,  
               ]

# some challenging ones
# three fingers
hand_poses_3 = [0.09, 0, 0.02,
                3, 1.5, 0.5,
                0.349,  0.975,  0.369,   0.172,
                0.0559,  1.19,  0.,   0.613,
                -0.213,  1.57,  1.57,   1.57, 
                0,  -0.349,  1.57,   1.57,   1.57, 
                -0.0209,  1.22,  -0.0963,   -0.112,   -0.554,  
               ]

H2O_PRESET = [hand_poses_0, hand_poses_1, hand_poses_2, hand_poses_3]


grasp_code_selected = [
'core-bottle-11fc9827d6b467467d3aa3bae1f7b494',
'core-bowl-1b4d7803a3298f8477bdcb8816a3fac9',
'core-cellphone-521fa79c95f4d3e26d9f55fbf45cc0c',
'sem-Vase-6af347c8e74b5b715d878ba9ec3c0d6a',
'mujoco-UGG_Cambridge_Womens_Black_7',
'core-mug-6dd59cc1130a426571215a0b56898e5e',
]

class DexGraspHandToObjectData(Dataset):
    def __init__(self,
                 cfg,
                mode,
                gen_object_num,
                source='preset',
                data=None,
                **kwargs,
                ):
        super().__init__()
        self.mode = 'test'

        self.data_dir = cfg.DATA.DATA_DIR

        self.rot_type = cfg.DATA.ROT_TYPE
        logger.info("dataset.rot_type: {}", self.rot_type)
        assert self.rot_type == 'mat'
        
        if source == 'preset':
            self.hand_poses, self.data_names = self.set_data()
        elif source == 'dataset':
            self.hand_poses, self.data_names = self.set_data_by_dataset()
        elif source == 'manual':
            assert isinstance(data, list)
            if not isinstance(data[0], list):
                data = [data]
            self.hand_poses, self.data_names = self.set_data(data)

        self.dataset_length = len(self.hand_poses)
        logger.info('initial dataset_lenght: {}', self.dataset_length)
        self.gen_object_num = gen_object_num
        
    def set_data_by_dataset(self):
        hand_poses = []
        data_names = []
        for grasp_code in grasp_code_selected:
            grasp_data = np.load(os.path.join(self.data_dir, grasp_code + '.npy'), allow_pickle=True)
            for idx in range(50):
                grasp_idx = grasp_data[idx]
                qpos = grasp_idx['qpos']
                translation = [qpos[name] for name in translation_names]
                rotation = transforms3d.euler.euler2mat(*[qpos[name] for name in rot_names])
                rotation = rotation[:, :2].T.ravel().tolist()
                joint_angles = [qpos[name] for name in joint_names], np.float32
                hand_pose = translation + rotation + joint_angles
                hand_poses.append(hand_pose)
                data_names.append(f"{grasp_code}__{idx}#{idx}")
        return hand_poses, data_names

    def set_data(self, data=None):
        hand_poses = []
        data_names = []
        if data is None:
            data = H2O_PRESET
            data_name = 'preset'
        else:
            data_name = 'manual'
        for i in range(len(data)):
            d_i = data[i]
            rotation = transforms3d.euler.euler2mat(*d_i[3:6])
            rotation = rotation[:, :2].T.ravel().tolist()
            hand_pose = d_i[:3] + rotation + d_i[6:]
            hand_poses.append(hand_pose)
            data_names.append(f"{data_name}_{i}#0")
        return hand_poses, data_names

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        hand_pose = self.hand_poses[idx]
        data_name = self.data_names[idx]
        
        hand_pose = torch.tensor(hand_pose, dtype=torch.float32)
        ori_hand_pose = hand_pose.clone()
        hand_t, hand_R, hand_param = decompose_hand_param(hand_pose, rot_type=self.rot_type, normalize=True)
        hand_pose = torch.cat([hand_t, hand_R, hand_param], dim=-1)
        
        ori_hand_pose = ori_hand_pose.unsqueeze(0).repeat(self.gen_object_num, 1)
        hand_pose = hand_pose.unsqueeze(0).repeat(self.gen_object_num, 1)

        data_dict = {
            'ori_hand_pose': ori_hand_pose,
            'hand_pose': hand_pose,
            'grasp_code': data_name,
        }

        return data_dict


def build_h2o_dataloader(cfg, source='preset', **kwargs):
    data_dict = dict(
        cfg=cfg,
        mode="test",
        gen_object_num=cfg.GEN_OBJECT_NUM,
        source=source
    )
    data_dict.update(kwargs)

    test_set = DexGraspHandToObjectData(**data_dict)
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.NUM_WORKERS > 0)
    )
    return None, test_loader
