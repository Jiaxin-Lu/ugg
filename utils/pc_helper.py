import torch
import numpy as np
import open3d as o3d


def save_pcd(dir, pc, color=None):
    if isinstance(pc, torch.Tensor):
        pc = pc.detach().cpu().numpy()
    xyz = o3d.geometry.PointCloud()
    xyz.points = o3d.utility.Vector3dVector(pc)
    if color is not None:
        if isinstance(color, torch.Tensor):
            color = color.detach().cpu().numpy()
        xyz.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_point_cloud(dir, xyz)
