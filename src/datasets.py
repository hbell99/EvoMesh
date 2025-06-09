import os
import pickle
import numpy as np
import h5py
import torch
from torch_geometric.data import Data, Dataset
import json
from enum import Enum
from helpers_mesh import tetras_to_edges, triangles_to_edges, quads_to_edges, lines_to_edges
from helpers_bistride import generate_multi_layer_stride, SeedingHeuristic

class MeshType(Enum):
    Triangle = 1
    Tetrahedron = 2
    Quad = 3
    Line = 4
    Flat = 5


class MeshGeneralDataset(Dataset):
    def __init__(self,
                 root,
                 in_normal_feature_list,
                 out_normal_feature_list,
                 roll_normal_feature_list,
                 instance_id,
                 layer_num,
                 stride,
                 mode,
                 noise_shuffle,
                 noise_level,
                 noise_gamma,
                 recal_mesh,
                 consist_mesh,
                 mesh_type,
                 has_contact,
                 has_self_contact,
                 dirichelet_markers=[],
                 must_preserve_type=[],
                 save_cells=True,
                 refine_steps=[],
                 condition_steps=0,
                 seed_heuristic=SeedingHeuristic.MinAve):
        # NOTE instance_id for a specific transient seq; each instance is in shape T,N,F
        self.instance_id = instance_id
        self.mode = mode
        self.data_dir = os.path.join(root, 'outputs_' + mode + '/')
        self.layer_num = layer_num
        self.recal_mesh = recal_mesh
        self.consist_mesh = consist_mesh
        self.mesh_type = mesh_type
        self.has_contact = has_contact
        self.has_self_contact = has_self_contact
        self.dirichelet_markers = dirichelet_markers
        self.seed_heuristic = seed_heuristic
        self.save_cells = save_cells
        self.refine_steps = refine_steps
        self.condition_steps = condition_steps
        self.prediction_steps = 0
        # read all features indicated in meta
        with open(os.path.join(root, 'meta.json'), 'r') as fp:
            self.meta = json.loads(fp.read())
        field_names = self.meta['field_names']
        fields = dict()
        with h5py.File(os.path.join(self.data_dir, str(instance_id) + '.h5'), 'r') as f:
            for name in field_names:
                if name == "cells":
                    fields[name] = np.array(f[name])
                    self.cells = fields[name][0]
                else:
                    fields[name] = torch.tensor(np.array(f[name]), dtype=torch.float)
        
        # read normalization info
        self._read_normalization_info(in_normal_feature_list, out_normal_feature_list, roll_normal_feature_list)
        # shuffle noise and enhance data, determine do or not
        if noise_level is None or not noise_shuffle:
            self.noise_shuffle = False
            self.noise_shuffle = None
            self.noise_gamma = 1.0
        else:
            self.noise_shuffle = True
            self.noise_level = torch.tensor(noise_level, dtype=torch.float)
            self.noise_gamma = noise_gamma
        # cal len of dataset
        self.stride = stride
        self.strided_idx = list(range(0, fields["mesh_pos"].shape[0], stride))
        self.L = len(self.strided_idx) - 1
        for name in field_names:
            fields[name] = fields[name][self.strided_idx]
        in_feature, tar_feature = self._preprocess(fields)
        # normalization
        self.in_feature, self.tar_feature = self._normalize(in_feature, tar_feature)
        #
        self._cal_multi_mesh(fields)

        self.nodes = list(range(self.in_feature.shape[1]))
        self.must_preserve_type = must_preserve_type
        super().__init__(root)

    def len(self):
        return self.L - (self.condition_steps + self.prediction_steps)

    def get(self, idx):
        # idx in time seq (enhanced by noise shuffle)
        # also return the midx and mgs, for combining
        if self.condition_steps > 0:
            condition_feat = self.in_feature[idx: idx+self.condition_steps]
            condition_feat = condition_feat.transpose(0, 1)
        else:
            condition_feat = self.in_feature[idx]
        if self.prediction_steps > 0:
            prediction_feat = self.tar_feature[idx+self.condition_steps: idx+self.condition_steps+self.prediction_steps]
            prediction_feat = prediction_feat.transpose(0, 1)
        else:
            prediction_feat = self.tar_feature[idx+self.condition_steps]
        if self.condition_steps > 0:
            t = torch.arange(idx, idx+self.condition_steps) / (self.L + 1)
            data = Data(x=condition_feat, y=prediction_feat, t=t)
        else:
            data = Data(x=condition_feat, y=prediction_feat, t=idx)
        return data

    def _read_normalization_info(self, in_normal_feature_list, out_normal_feature_list, roll_normal_feature_list):
        # collect in normalization
        for i, fea in enumerate(in_normal_feature_list):
            temp_std = torch.tensor(self.meta['normalization_info'][fea]['std'], dtype=torch.float)
            temp_mean = torch.tensor(self.meta['normalization_info'][fea]['mean'], dtype=torch.float)
            if i == 0:
                self.std_in = temp_std
                self.mean_in = temp_mean
            else:
                self.std_in = torch.cat((self.std_in, temp_std), dim=-1)
                self.mean_in = torch.cat((self.mean_in, temp_mean), dim=-1)
        # collect out normalization
        for i, fea in enumerate(out_normal_feature_list):
            temp_std = torch.tensor(self.meta['normalization_info'][fea]['std'], dtype=torch.float)
            temp_mean = torch.tensor(self.meta['normalization_info'][fea]['mean'], dtype=torch.float)
            if i == 0:
                self.std_out = temp_std
                self.mean_out = temp_mean
            else:
                self.std_out = torch.cat((self.std_out, temp_std), dim=-1)
                self.mean_out = torch.cat((self.mean_out, temp_mean), dim=-1)
        # collect roll-out normalization
        self.roll_l = 0
        for i, fea in enumerate(roll_normal_feature_list):
            temp_std = torch.tensor(self.meta['normalization_info'][fea]['std'], dtype=torch.float)
            self.roll_l += temp_std.shape[-1]
        # NOTE assume/let all leading features align with the list ordering here
        self.in_norm_l = self.std_in.shape[0]
        self.out_norm_l = self.std_out.shape[0]

    def _normalize(self, t_in, t_out):
        x_in = t_in.clone()
        x_out = t_out.clone()
        x_in[..., :self.in_norm_l] = (x_in[..., :self.in_norm_l] - self.mean_in) / self.std_in
        x_out[..., :self.out_norm_l] = (x_out[..., :self.out_norm_l] - self.mean_out) / self.std_out
        return x_in, x_out

    def _unnormalize(self, t_in, t_out):
        x_in = t_in.clone()
        x_out = t_out.clone()
        x_in[..., :self.in_norm_l] = x_in[..., :self.in_norm_l] * self.std_in + self.mean_in
        x_out[..., :self.in_norm_l] = x_out[..., :self.in_norm_l] * self.std_out + self.mean_out
        return x_in, x_out

    def _push_forward(self, out, current_stat):
        output_stat = torch.zeros_like(current_stat)
        output_stat[..., :self.roll_l] = out[..., :self.roll_l]
        output_stat[..., self.roll_l:] = current_stat[..., self.roll_l:].detach()
        return output_stat
        
        # current_stat[..., :self.roll_l] = out[..., :self.roll_l]
        # return current_stat

    def suggested_pen_coef(self):
        return self.std_out * self.std_out

    def _preprocess(self, fields):
        # NOTE implement in child class
        raise NotImplementedError("This needs to be implemented")

    def _cal_multi_mesh(self, fields):
        if not self.has_contact:
            if self.consist_mesh:
                mmfile = os.path.join(self.data_dir, 'mmesh_layer_' + str(self.layer_num) + '.dat')
            else:
                mmfile = os.path.join(self.data_dir, str(self.instance_id) + '_mmesh_layer_' + str(self.layer_num) + '.dat')
            mmexist = os.path.isfile(mmfile)
            if self.recal_mesh or not mmexist:
                if self.mesh_type == MeshType.Triangle:
                    edge_i = triangles_to_edges(self.cells)
                if self.mesh_type == MeshType.Tetrahedron:
                    edge_i = tetras_to_edges(self.cells)
                if self.mesh_type == MeshType.Quad:
                    edge_i = quads_to_edges(self.cells)
                if self.mesh_type == MeshType.Line:
                    edge_i = lines_to_edges(self.cells)
                if self.mesh_type == MeshType.Flat:
                    edge_i = self.cells
                m_gs, m_ids = generate_multi_layer_stride(edge_i,
                                                          self.layer_num,
                                                          seed_heuristic=self.seed_heuristic,
                                                          n=fields['mesh_pos'].shape[-2],
                                                          pos_mesh=fields["mesh_pos"][0].clone().detach().numpy())
                m_mesh = {'m_gs': m_gs, 'm_ids': m_ids}
                pickle.dump(m_mesh, open(mmfile, 'wb'))
            else:
                m_mesh = pickle.load(open(mmfile, 'rb'))
                m_gs, m_ids = m_mesh['m_gs'], m_mesh['m_ids']
            self.m_g = m_gs
            self.m_idx = m_ids
        else:
            raise NotImplementedError("Contact mesh generation is not implemented yet")

        self.recal_mesh = False


class MeshCylinderDataset(MeshGeneralDataset):
    def __init__(self, root, instance_id, layer_num, stride, mode='train', noise_shuffle=False, noise_level=None, noise_gamma=1.0, 
                 refine_steps=[], recal_mesh=False, consist_mesh=False, step=-1, condition_steps=0, args=None):
        in_normal_feature_list, out_normal_feature_list, roll_normal_feature_list = ['velocity', 'mesh_pos'], ['velocity'], ['velocity']
        self.step = step
        self.args = args
        super().__init__(root,
                         in_normal_feature_list,
                         out_normal_feature_list,
                         roll_normal_feature_list,
                         instance_id,
                         layer_num,
                         stride,
                         mode,
                         noise_shuffle,
                         noise_level,
                         noise_gamma,
                         recal_mesh,
                         consist_mesh,
                         mesh_type=MeshType.Triangle,
                         has_contact=False,
                         has_self_contact=False,
                         dirichelet_markers=[1, 2, 3, 4],
                         refine_steps=refine_steps,
                         must_preserve_type=[4, 6],
                         condition_steps=condition_steps,
                         seed_heuristic=SeedingHeuristic.MinAve)

    def _preprocess(self, fields):
        # noise shuffle
        # in: vel, type
        # out: d_vel
        node_info_inp = fields["velocity"][:-1].clone()
        node_info_tar = fields["velocity"][1:].clone()
        # enhance by noise level
        if self.noise_shuffle:
            # collect special nodes
            node_type = fields["node_type"][:-1]
            preset_node = ((node_type != 0) * (node_type != 5)).bool()
            no_noise_node = preset_node
            # collect special nodes
            noise_base = torch.ones_like(node_info_tar)
            noise_base[:, :] = self.noise_level
            noise = torch.normal(0.0, noise_base)
            # for dirichelet nodes, the noise is zero
            noise = torch.where(no_noise_node, torch.zeros_like(noise), noise)
            node_info_inp += noise
            node_info_tar += (1.0 - self.noise_gamma) * noise
        node_info_inp = torch.cat((node_info_inp, fields["mesh_pos"][:-1], fields["node_type"][:-1]), dim=-1)
        return node_info_inp, node_info_tar


class MeshFlagDataset(MeshGeneralDataset):
    def __init__(self, root, instance_id, layer_num, stride, mode='train', noise_shuffle=False, noise_level=None, noise_gamma=1.0, 
                 refine_steps=[], recal_mesh=False, consist_mesh=False, args=None):
        in_normal_feature_list, out_normal_feature_list, roll_normal_feature_list = ['world_pos', 'mesh_pos'], ['world_pos'], ['world_pos']
        super().__init__(root,
                         in_normal_feature_list,
                         out_normal_feature_list,
                         roll_normal_feature_list,
                         instance_id,
                         layer_num,
                         stride,
                         mode,
                         noise_shuffle,
                         noise_level,
                         noise_gamma,
                         recal_mesh,
                         consist_mesh,
                         mesh_type=MeshType.Triangle,
                         has_contact=False,
                         has_self_contact=False,
                         dirichelet_markers=[],
                         refine_steps=refine_steps, 
                         must_preserve_type=[3],)
        self.L = self.L - 1

    def _preprocess(self, fields):
        node_info_his = fields["world_pos"][:-2].clone()
        node_info_inp = fields["world_pos"][1:-1].clone()
        node_info_tar = fields["world_pos"][2:].clone()
        # enhance by noise level
        if self.noise_shuffle:
            # collect special nodes
            node_type = fields["node_type"][1:-1]
            no_noise_node = (node_type == 3).bool()
            noise_base = torch.ones_like(node_info_tar)
            noise_base[:, :] = self.noise_level
            noise = torch.normal(0.0, noise_base)
            # for dirichelet nodes, the noise is zero
            noise = torch.where(no_noise_node, torch.zeros_like(noise), noise)
            node_info_inp += noise
            node_info_tar += (1.0 - self.noise_gamma) * noise
        node_info_vel = node_info_inp - node_info_his
        mesh_pos = torch.cat((fields["mesh_pos"], torch.zeros_like(fields["mesh_pos"][..., :1])), dim=-1)
        node_info_inp = torch.cat((node_info_inp, mesh_pos[1:-1], node_info_vel, fields["node_type"][1:-1]), dim=-1)
        return node_info_inp, node_info_tar
