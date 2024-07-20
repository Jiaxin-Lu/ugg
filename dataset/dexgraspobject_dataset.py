import os
import pickle
import random
import numpy as np
import torch
import trimesh as tm
from torch.utils.data import Dataset, DataLoader
from loguru import logger


class DexGraspObjectDataset(Dataset):
    def __init__(self, 
                 cfg,
                 mode,  

                 gen_grasp_num=50,
                 test_range=[0, -1],
                 ):
        self.mode = mode
        self.data_dir = cfg.DATA.DATA_DIR
        self.mesh_dir = cfg.DATA.MESH_DIR
        self.splits_dir = cfg.DATA.SPLITS_DIR
        self.pc_num_points = cfg.DATA.PC_NUM_POINTS
        self.scale_choice = torch.tensor([0.06, 0.08, 0.1, 0.12, 0.15], dtype=torch.float32)
        self.each_scale_gen_grasp_num = gen_grasp_num
        self.gen_grasp_num = self.scale_choice.shape[0] * self.each_scale_gen_grasp_num

        self.normalize_pc=cfg.DATA.NORMALIZE_PC

        self.mesh_code_list, self.mesh_dir_list = self.read_data(self.mesh_dir, self.splits_dir, mode)
        
        self.dataset_length = len(self.mesh_code_list)

        st = test_range[0]
        ed = test_range[1] if test_range[1] > 0 else self.dataset_length
        self.mesh_code_list = [self.mesh_code_list[i] for i in range(st, ed)]
        self.mesh_dir_list = [self.mesh_dir_list[i] for i in range(st, ed)]
        self.dataset_length = len(self.mesh_code_list)
        logger.info(f"update dataset range to {st}:{ed}")


    def read_data(self, mesh_dir, splits_dir, mode):
        pk_file_name = f"saved_dexgraspobject_{mode}"
        pk_file_name += '.pk'
        if os.path.exists(os.path.join(splits_dir, pk_file_name)):
            print('load from existing presaved file')
            with open(os.path.join(splits_dir, pk_file_name), 'rb') as f:
                data_dict = pickle.load(f)
            return data_dict['mesh_code_list'], data_dict['mesh_dir_list']
        
        print("generate presaved files")
        grasp_code_list = []
        with open(os.path.join(splits_dir, f'split_dexgraspobject_{mode}.txt'), 'r') as f:
            grasp_code_list += f.read().splitlines()
        grasp_code_list = [grasp_code for grasp_code in grasp_code_list if len(grasp_code) > 1]
        grasp_code_list.sort()
        
        mesh_dir_list = []
        for grasp_code in grasp_code_list:
            mesh_obj_path = os.path.join(mesh_dir, grasp_code, "coacd/decomposed.obj")
            mesh_dir_list.append(mesh_obj_path)
        data_dict = {
            'mesh_code_list': grasp_code_list,
            'mesh_dir_list': mesh_dir_list,
        }
        print("finish, saving...")
        with open(os.path.join(splits_dir, pk_file_name), 'wb') as f:
            pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        return grasp_code_list, mesh_dir_list
            
    def __len__(self):
        return self.dataset_length
    
    def __getitem__(self, idx):
        mesh_code = self.mesh_code_list[idx]
        mesh_obj_path = self.mesh_dir_list[idx]
        mesh = tm.load(mesh_obj_path, force='mesh')
        samples, fid = mesh.sample(self.pc_num_points, return_index=True)
        ori_pc = torch.tensor(samples, dtype=torch.float)
        object_pc = torch.tensor(samples, dtype=torch.float).unsqueeze(0).repeat(self.gen_grasp_num, 1, 1)
        object_sc = torch.repeat_interleave(self.scale_choice, self.each_scale_gen_grasp_num).reshape(-1, 1)
        object_pc = object_pc * object_sc.reshape(-1, 1, 1)
        if self.normalize_pc:
            object_pc = object_pc * 6.6
        data_dict = {
            'object_pc': object_pc,
            'object_sc': object_sc,
            'grasp_code': mesh_code + "#-1",
            'ori_pc': ori_pc,
        }
        return data_dict
    

def build_dexgraspobject_dataloader(cfg):
    data_dict = dict(
        cfg=cfg,
        mode='train',
        test_range=cfg.DATA.RANGE,
        gen_grasp_num=cfg.GEN_GRASP_NUM,
    )

    data_dict['mode'] = 'test'
    test_set = DexGraspObjectDataset(**data_dict)
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.NUM_WORKERS > 0)
    )
    # no training for now
    return None, test_loader
