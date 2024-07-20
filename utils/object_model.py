import os

import numpy as np
import torch
import trimesh as tm
from torchsdf import compute_sdf, index_vertices_by_faces

SCALE_CHOICE = [0.06, 0.08, 0.1, 0.12, 0.15]

class ObjectModel:
    def __init__(self, data_dir, mesh_dir, pc_num_points, device='cpu', dataset_name='dex'):
        self.data_dir = data_dir
        self.mesh_dir = mesh_dir
        self.device = device
        self.pc_num_points = pc_num_points

        self.object_code_list = None
        self.object_mesh_list = None
        self.object_face_verts_list = None
        self.object_pc = None
        self.object_sc = None
        self.num_obj = None
        self.batch_size_each = 1
        self.dataset_name = dataset_name

    def set_parameters(self, object_codes, object_pc=None, object_sc=None, object_rot=None):
        if not isinstance(object_codes, list):
            object_codes = [object_codes]

        self.object_code_list = []
        for code in object_codes:
            code_tmp = str(code).split("#")
            idx = int(code_tmp[-1])
            object_code = code_tmp[0]
            self.object_code_list.append((object_code, idx))
        
        self.object_mesh_list = []
        self.object_face_verts_list = []
        surface_points_tensor = []
        scale_tensor = []
        for i, (object_code, idx) in enumerate(self.object_code_list):
            mesh = tm.load(os.path.join(self.mesh_dir, object_code, "coacd", "decomposed.obj"), force="mesh", process=False)
            self.object_mesh_list.append(mesh)
            object_verts = torch.Tensor(mesh.vertices).to(self.device)
            object_faces = torch.Tensor(mesh.faces).long().to(self.device)
            self.object_face_verts_list.append(index_vertices_by_faces(object_verts, object_faces).to(torch.double))
            if object_pc is None and self.pc_num_points != 0:
                samples, fid = mesh.sample(self.pc_num_points, return_index=True)
                surface_points = torch.tensor(samples, dtype=torch.float, device=self.device)
                surface_points_tensor.append(surface_points)
            if object_sc is None:
                grasp_data = np.load(os.path.join(self.data_dir, object_code+".npy"), allow_pickle=True)[idx]
                scale_tensor.append(grasp_data['scale'])
        
        if object_pc is None and self.pc_num_points != 0:
            self.object_pc = torch.stack(surface_points_tensor, dim=0).to(self.device)
        elif self.pc_num_points != 0:
            self.object_pc = object_pc.to(self.device)
        
        if object_sc is None:
            self.object_sc = torch.tensor(scale_tensor, dtype=torch.float32).to(self.device)
        else:
            self.object_sc = object_sc.to(self.device)

        self.num_obj = len(self.object_mesh_list)
        if self.num_obj > 1:
            assert self.num_obj == self.object_pc.shape[0]
            assert self.object_pc.shape[0] == self.object_sc.shape[0]
        elif self.num_obj == 1 and self.object_pc.shape[0] > 1:
            self.batch_size_each = self.object_pc.shape[0]
            assert self.object_pc.shape[0] == self.object_sc.shape[0]
        elif self.num_obj == 1 and self.object_sc.shape[0] > 1:
            self.batch_size_each = self.object_sc.shape[0]
            self.object_pc = self.object_pc.repeat(self.object_sc.shape[0], 1, 1) * self.object_sc.reshape(-1, 1, 1)
        
        if object_rot is not None:
            assert self.object_pc.shape[0] == object_rot.shape[0]
            self.object_pc = torch.bmm(self.object_pc, object_rot)
    
    def extend(self):
        self.object_pc = self.object_pc.repeat(2, 1, 1)
        self.object_sc = self.object_sc.repeat(2, 1)
        if self.num_obj > 1:
            self.object_code_list.extend(self.object_code_list)
            self.object_mesh_list.extend(self.object_mesh_list)
            self.object_face_verts_list.extend(self.object_face_verts_list)
            self.num_obj = len(self.object_mesh_list)
        else:
            self.batch_size_each = self.object_sc.shape[0]
            
    def cal_distance(self, x, with_closest_points=False):
        """
        Calculate signed distances from hand contact points to object meshes and return contact normals
        
        Interiors are positive, exteriors are negative
        
        Use our modified Kaolin package
        
        Parameters
        ----------
        x: (B, `n_contact`, 3) torch.Tensor
            hand contact points
        with_closest_points: bool
            whether to return closest points on object meshes
        
        Returns
        -------
        distance: (B, `n_contact`) torch.Tensor
            signed distances from hand contact points to object meshes, inside is positive
        normals: (B, `n_contact`, 3) torch.Tensor
            contact normal vectors defined by gradient
        closest_points: (B, `n_contact`, 3) torch.Tensor
            contact points on object meshes, returned only when `with_closest_points is True`
        """
        _, n_points, _ = x.shape
        x = x.reshape(-1, self.batch_size_each * n_points, 3).to(torch.double)
        distance = []
        normals = []
        closest_points = []
        scale = self.object_sc.reshape(-1, self.batch_size_each).repeat_interleave(n_points, dim=1)
        x = x / scale.unsqueeze(2)
        for i in range(self.num_obj):
            face_verts = self.object_face_verts_list[i]
            dis, dis_signs, normal, _ = compute_sdf(x[i], face_verts)
            if with_closest_points:
                closest_points.append(x[i] - dis.sqrt().unsqueeze(1) * normal)
            dis = torch.sqrt(dis + 1e-8)
            dis = dis * (-dis_signs)
            distance.append(dis)
            normals.append(normal * dis_signs.unsqueeze(1))
        distance = torch.stack(distance)
        normals = torch.stack(normals)
        distance = distance * scale
        distance = distance.reshape(-1, n_points)
        normals = normals.reshape(-1, n_points, 3)
        if with_closest_points:
            closest_points = (torch.stack(closest_points) * scale.unsqueeze(2)).reshape(-1, n_points, 3)
            return distance, normals, closest_points
        return distance, normals
    