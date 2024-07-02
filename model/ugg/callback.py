import copy
import os
import pickle

import numpy as np
import open3d as o3d
import torch
from pytorch_lightning import Callback
from pytorch_lightning.strategies import DDPStrategy

from utils.hand_helper import compose_hand_param


class DiffusionVisCallback(Callback):
    def __init__(self, cfg, sample_num, train_loader, test_loader, task=['joint', 'obj2hand', 'hand2obj']):
        super().__init__()

        self.save_path = os.path.join(cfg.OUTPUT_PATH, 'eval_save')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        self.sample_num = sample_num
        self.task = task
        self.train_loader = train_loader
        self.test_loader = test_loader
    
    def _sample_vis(self, trainer, pl_module, dataloader, mode, current_save_path):
        if not self.task:
            return
        num = self.sample_num
        batch = next(iter(dataloader))
                
        ret_dict = pl_module.sample(data_dict=batch, num=num, task=self.task)
        
        for task in ret_dict.keys():
            vis_dict = ret_dict[task]
            task_save_path = os.path.join(current_save_path, f"{task}_{mode}")
            os.makedirs(task_save_path, exist_ok=True)
            
            pc = vis_dict['pc_final'].detach().cpu().numpy()
            hand_t = vis_dict['hand_t_final'].detach().cpu().numpy()
            hand_R = vis_dict['hand_R_final'].detach().cpu().numpy()
            hand_param = vis_dict['hand_param_final'].detach().cpu().numpy()
            
            contact_map = None
            contact_map_ori = None
            hand_pts = None
            if "contact_map_final" in vis_dict and vis_dict['contact_map_final'] is not None:
                contact_map = vis_dict['contact_map_final'].detach().cpu().numpy()
                contact_map_ori = vis_dict['contact_map_ori'].detach().cpu().numpy()
                contact_map_gt = vis_dict['contact_map_gt'].detach().cpu().numpy()
            if 'hand_pts' in vis_dict and vis_dict['hand_pts'] is not None:
                hand_pts = vis_dict['hand_pts'].detach().cpu().numpy()

            hand_param_total = compose_hand_param(hand_t, hand_R, hand_param, rot_type=pl_module.hand_rot_type)

            for b in range(num):
                pc_b = pc[b][:, :3]
                xyz = o3d.geometry.PointCloud()
                xyz.points = o3d.utility.Vector3dVector(pc_b)
                if contact_map is not None:
                    pc_new = np.concatenate([pc_b, contact_map[b], contact_map_gt[b]], axis=0)
                    color_contact = np.ones_like(contact_map[b])
                    color_contact_gt = np.ones_like(contact_map_gt[b])
                    color_contact[:, 0] = 0
                    color_contact_gt[:, 1] = 0
                    # xyz.colors = o3d.utility.Vector3dVector(color)

                    c = 0.5 + 0.5 * contact_map_ori[b]
                    c = c.reshape(-1, 1)
                    color = c * np.array([1., 0., 0.]) + (1. - c) * np.array([1., 1., 1.])
                    color = np.clip(color, 0, 1.)
                    color = np.concatenate([color, color_contact, color_contact_gt], axis=0)
                    if hand_pts is not None:
                        hand_pts_b = hand_pts[b] * 6.6
                        pc_new = np.concatenate([pc_new, hand_pts_b], axis=0)
                        c = np.zeros((hand_pts_b.shape[0], 3))
                        c[:, 2] = 1.
                        color = np.concatenate([color, c], axis=0)
                        
                    xyz.points = o3d.utility.Vector3dVector(pc_new)
                    xyz.colors = o3d.utility.Vector3dVector(color)
                o3d.io.write_point_cloud(os.path.join(task_save_path, f"pc_{b:03d}.ply"), xyz)
                np.savez(os.path.join(task_save_path, f"hand_{b:03d}.npz"), hand_param=hand_param_total[b], grasp_code=batch['grasp_code'][b])

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.callback_metrics.get("epoch", None)
        if isinstance(epoch, torch.Tensor):
            epoch = epoch.int().item()
        else:
            epoch = int(trainer.current_epoch)
        current_save_path = os.path.join(self.save_path, 'epoch_{:03d}'.format(epoch))
        if not os.path.exists(current_save_path):
            os.makedirs(current_save_path, exist_ok=True)

        pl_module.eval()
        self._sample_vis(trainer, pl_module, self.train_loader, 'train', current_save_path)
        self._sample_vis(trainer, pl_module, self.test_loader, 'val', current_save_path)

    @torch.no_grad()
    def on_test_epoch_end(self, trainer, pl_module) -> None:
        epoch = trainer.callback_metrics.get("epoch", None)
        if isinstance(epoch, torch.Tensor):
            epoch = epoch.int().item()
        else:
            epoch = int(trainer.current_epoch)
        current_save_path = os.path.join(self.save_path, 'test_{:03d}'.format(epoch))
        if not os.path.exists(current_save_path): 
            os.makedirs(current_save_path, exist_ok=True)

        pl_module.eval()
        self._sample_vis(trainer, pl_module, self.train_loader, 'train', current_save_path)
        self._sample_vis(trainer, pl_module, self.test_loader, 'val', current_save_path)


class CustomDDPStrategy(DDPStrategy):
    def configure_ddp(self):
        super().configure_ddp()
        self.model._set_static_graph()
