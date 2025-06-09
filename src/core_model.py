import torch.nn as nn
import torch
from torch.nn import Sequential as Seq, Linear, ReLU, LayerNorm, Softplus
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from torch_geometric.utils import add_remaining_self_loops, degree

from torch_geometric.utils import (
    remove_self_loops,
    to_edge_index,
    to_torch_csr_tensor,
    coalesce,
)


def gumbel_softmax_sample(logits, temperature):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
    y = logits + gumbel_noise
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=0.1, hard=False):
    y_soft = gumbel_softmax_sample(logits, temperature)
    
    # Straight through estimation: y_hard is like y, but gradients are like soft sample y
    shape = y_soft.size()
    _, max_idx = y_soft.max(dim=-1, keepdim=True)
    y_hard = torch.zeros_like(y_soft).scatter(-1, max_idx, 1.0)
    y_hard = y_hard - y_soft.detach() + y_soft
    
    if hard:
        y_soft = y_hard
    return y_soft, y_hard


class MLP(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, hidden_layers, layer_normalized=True):
        super(MLP, self).__init__()
        modules = []
        for l in range(hidden_layers):
            if l == 0:
                modules.append(Linear(input_dim, latent_dim))
            else:
                modules.append(Linear(latent_dim, latent_dim))
            modules.append(ReLU())
        modules.append(Linear(latent_dim, output_dim))
        if layer_normalized:
            modules.append(LayerNorm(output_dim, elementwise_affine=False))

        self.seq = Seq(*modules)

    def forward(self, x):
        return self.seq(x)


class amp_base(MessagePassing):
    def __init__(self, latent_dim, hidden_layer, pos_dim, lagrangian):
        super().__init__(aggr='add', flow='target_to_source')
        self.mlp_node_delta = MLP(2 * latent_dim, latent_dim, latent_dim, hidden_layer, True)
        edge_info_in_len = 2 * latent_dim + 2 * pos_dim + 2 if lagrangian else 2 * latent_dim + pos_dim + 1
        self.mlp_edge_info = MLP(edge_info_in_len, latent_dim, latent_dim, hidden_layer, True)
        self.mlp_edge_weight = Seq(*[MLP(latent_dim, latent_dim, 1, hidden_layer, False)])
        self.lagrangian = lagrangian
        self.pos_dim = pos_dim
        self.latent_dim = latent_dim

    def forward(self, x, g, pos):
        i = g[0]
        j = g[1]
        if len(x.shape) == 3:
            T, _, _ = x.shape
            x_i = x[:, i]
            x_j = x[:, j]
        elif len(x.shape) == 2:
            x_i = x[i]
            x_j = x[j]
        else:
            raise NotImplementedError("Only implemented for dim 2 and 3")
        if len(pos.shape) == 3:
            pi = pos[:, i]
            pj = pos[:, j]
        elif len(pos.shape) == 2:
            pi = pos[i]
            pj = pos[j]
        else:
            raise NotImplementedError("Only implemented for dim 2 and 3")
        dir = pi - pj  # in shape (T),N,dim
        if self.lagrangian:
            norm_w = torch.norm(dir[..., :self.pos_dim], dim=-1, keepdim=True)  # in shape (T),N,1
            norm_m = torch.norm(dir[..., self.pos_dim:], dim=-1, keepdim=True)  # in shape (T),N,1
            fiber = torch.cat([dir, norm_w, norm_m], dim=-1)
        else:
            norm = torch.norm(dir, dim=-1, keepdim=True)  # in shape (T),N,1
            fiber = torch.cat([dir, norm], dim=-1)

        if len(x.shape) == 3 and len(pos.shape) == 2:
            tmp = torch.cat([fiber.unsqueeze(0).repeat(T, 1, 1), x_i, x_j], dim=-1)
        else:
            tmp = torch.cat([fiber, x_i, x_j], dim=-1)

        edge_embedding = self.mlp_edge_info(tmp)

        edge_weight = self.mlp_edge_weight(edge_embedding)
        edge_weight = scatter_softmax(edge_weight, j, dim=-2)

        edge_embedding = edge_embedding * edge_weight

        aggr_out = scatter(edge_embedding, j, dim=-2, dim_size=x.shape[-2], reduce="sum")

        tmp = torch.cat([x, aggr_out], dim=-1)
        return self.mlp_node_delta(tmp) + x, edge_weight


class amp(MessagePassing):
    def __init__(self, latent_dim, hidden_layer, pos_dim, lagrangian):
        super().__init__(aggr='add', flow='target_to_source')
        self.mlp_node_delta = MLP(2 * latent_dim, latent_dim, latent_dim, hidden_layer, True)
        edge_info_in_len = 2 * latent_dim + 2 * pos_dim + 2 if lagrangian else 2 * latent_dim + pos_dim + 1
        self.weightnet = True
        if self.weightnet:
            self.mlp_edge_info = MLP(edge_info_in_len, latent_dim, latent_dim, hidden_layer, True)
            self.mlp_edge_weight = MLP(latent_dim, latent_dim, 1, hidden_layer, False)
        else:
            self.mlp_edge_info = MLP(edge_info_in_len, latent_dim, latent_dim + 1, hidden_layer, False)
        self.mlp_gumbel = Seq(*[MLP(2 * latent_dim, latent_dim, 2, hidden_layer, False)])
        self.lagrangian = lagrangian
        self.pos_dim = pos_dim
        self.latent_dim = latent_dim
    
    def forward(self, x, g, pos, temp):
        i = g[0]
        j = g[1]
        if len(x.shape) == 3:
            T, _, _ = x.shape
            x_i = x[:, i]
            x_j = x[:, j]
        elif len(x.shape) == 2:
            x_i = x[i]
            x_j = x[j]
        else:
            raise NotImplementedError("Only implemented for dim 2 and 3")
        if len(pos.shape) == 3:
            pi = pos[:, i]
            pj = pos[:, j]
        elif len(pos.shape) == 2:
            pi = pos[i]
            pj = pos[j]
        else:
            raise NotImplementedError("Only implemented for dim 2 and 3")
        dir = pi - pj  # in shape (T),N,dim
        if self.lagrangian:
            norm_w = torch.norm(dir[..., :self.pos_dim], dim=-1, keepdim=True)  # in shape (T),N,1
            norm_m = torch.norm(dir[..., self.pos_dim:], dim=-1, keepdim=True)  # in shape (T),N,1
            fiber = torch.cat([dir, norm_w, norm_m], dim=-1)
        else:
            norm = torch.norm(dir, dim=-1, keepdim=True)  # in shape (T),N,1
            fiber = torch.cat([dir, norm], dim=-1)
        
        if len(x.shape) == 3 and len(pos.shape) == 2:
            tmp = torch.cat([fiber.unsqueeze(0).repeat(T, 1, 1), x_i, x_j], dim=-1)
        else:
            tmp = torch.cat([fiber, x_i, x_j], dim=-1)
        
        edge_embedding = self.mlp_edge_info(tmp)
        if self.weightnet:
            edge_weight = self.mlp_edge_weight(edge_embedding)
        else:
            edge_embedding, edge_weight = edge_embedding[..., :-1], edge_embedding[..., [-1]]
        edge_weight = scatter_softmax(edge_weight, j, dim=-2)

        edge_embedding = edge_embedding * edge_weight

        aggr_out = scatter(edge_embedding, j, dim=-2, dim_size=x.shape[-2], reduce="sum")

        tmp = torch.cat([x, aggr_out], dim=-1)

        logits = self.mlp_gumbel(tmp)
        logits = torch.mean(logits, axis=0)
        
        hard = True
        y_soft, y_hard = gumbel_softmax(logits, temp, hard=hard)  # tau越小越接近one-hot

        return self.mlp_node_delta(tmp) + x, edge_weight, y_hard


class WeightedEdgeConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add', flow='target_to_source')

    def forward(self, x, g, ew, aggragating=True):
        # aggregating: False means returning
        i = g[0]
        j = g[1]
        if len(x.shape) == 3:
            weighted_info = x[:, i] if aggragating else x[:, j]
        elif len(x.shape) == 2:
            weighted_info = x[i] if aggragating else x[j]
        else:
            raise NotImplementedError("Only implemented for dim 2 and 3")
        if len(ew.shape) == len(x.shape):
            weighted_info = weighted_info * ew
        else:
            weighted_info *= ew.unsqueeze(-1)
        target_index = j if aggragating else i
        aggr_out = scatter(weighted_info, target_index, dim=-2, dim_size=x.shape[-2], reduce="sum")
        return aggr_out

    @torch.no_grad()
    def cal_ew(self, w, g):
        deg = degree(g[0], dtype=torch.float, num_nodes=w.shape[0])
        normed_w = w.squeeze(-1) / deg
        i = g[0]
        j = g[1]
        w_to_send = normed_w[i]
        eps = 1e-12
        aggr_w = scatter(w_to_send, j, dim=-1, dim_size=normed_w.size(0), reduce="sum") + eps
        ec = w_to_send / aggr_w[j]
        return ec, aggr_w


class Unpool(nn.Module):
    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, h, pre_node_num, idx):
        if len(h.shape) == 2:
            new_h = h.new_zeros([pre_node_num, h.shape[-1]])
            new_h[idx] = h
        elif len(h.shape) == 3:
            new_h = h.new_zeros([h.shape[0], pre_node_num, h.shape[-1]])
            new_h[:, idx] = h
        return new_h


class EvoMesh(nn.Module):
    def __init__(self, l_n, pre_l_n, bottom_ln, ld, hidden_layer, pos_dim, lagrangian, enhance=True, agg_conv_pos=False, edge_set_num=1):
        super(EvoMesh, self).__init__()
        self.down_gmps = nn.ModuleList()
        self.up_gmps = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.l_n = l_n
        self.edge_conv = WeightedEdgeConv()
        self.pre_l_n = pre_l_n
        self.enhance = enhance
        self.agg_conv_pos = agg_conv_pos
        self.bottom_ln = bottom_ln
        self.bottom_gmp = nn.ModuleList(amp_base(ld, hidden_layer, pos_dim, lagrangian) for _ in range(self.bottom_ln))
        for _ in range(self.l_n):
            if _ < self.pre_l_n:
                self.down_gmps.append(amp_base(ld, hidden_layer, pos_dim, lagrangian))
            else:
                self.down_gmps.append(amp(ld, hidden_layer, pos_dim, lagrangian))
            self.up_gmps.append(amp_base(ld, hidden_layer, pos_dim, lagrangian))
            self.unpools.append(Unpool())
        self.esn = edge_set_num
        self.lagrangian = lagrangian
        
    def pool_edge(self, g, idx, num_nodes):
        idx = idx.to(torch.long)
        idx_new_valid = torch.arange(len(idx), dtype=torch.long, device=g.device)
        idx_new_all = -1 * torch.ones(num_nodes, dtype=torch.long, device=g.device)
        idx_new_all[idx] = idx_new_valid
        new_g = -1 * torch.ones_like(g, dtype=torch.long, device=g.device)
        new_g[0] = idx_new_all[g[0]]
        new_g[1] = idx_new_all[g[1]]
        both_valid = (new_g[0] >= 0) & (new_g[1] >= 0)
        e_idx = torch.where(both_valid)[0]
        new_g = new_g[:, e_idx]

        return new_g

    def _pool_tensor(self, tensor, m_ids, y_hard, i, is_pre_layer):
        if is_pre_layer:
            return tensor[:, m_ids[i]] if len(tensor.shape) == 3 else tensor[m_ids[i]]
        
        mask = y_hard[..., 0].bool()
        tensor = y_hard[..., 0].unsqueeze(-1) * tensor
        return tensor[:, mask, :] if len(tensor.shape) == 3 else tensor[mask, :]

    def forward(self, h, mm_ids, mm_gs, pos, temp=0.1, weights=None):
        # h is in shape of (T), N, F
        # if edge_set_num>1, then m_g is in shape: Level,(Set),2,Edges, the 0th Set is main/material graph
        # pos is in (T),N,D
        down_outs = []
        down_ps = []
        cts = []
        w = pos.new_ones((pos.shape[-2], 1)) if weights is None else weights
        
        m_ids = mm_ids[:self.pre_l_n]
        m_gs = mm_gs[:self.pre_l_n + 1]
        # down pass
        l_n = self.l_n 
        for i in range(l_n):
            num_nodes = h.shape[-2] if i == 0 else len(m_ids[i-1]) #.shape[0]
            if self.esn > 1:
                gs = []
                gs_main, _ = add_remaining_self_loops(m_gs[i][0])
                gs_cont, _ = add_remaining_self_loops(m_gs[i][1]) 
                gs = [gs_main, gs_cont]
            else:
                gs, _ = add_remaining_self_loops(m_gs[i]) 
            if i < self.pre_l_n:
                h, ew = self.down_gmps[i](h, gs, pos)
                if i == 0 and self.lagrangian:
                    h, ew = self.down_gmps[i](h, gs, pos)
                y_hard = None
            else:
                h, ew, y_hard = self.down_gmps[i](h, gs, pos, temp)
                if i == 0 and self.lagrangian:
                    h, ew, y_hard = self.down_gmps[i](h, gs, pos, temp)
                N = h.shape[1]
                edge_index = m_gs[i]
                if self.enhance:
                    adj = to_torch_csr_tensor(edge_index, size=(N, N))
                    edge_index2, _ = to_edge_index(adj @ adj)
                    edge_index2, _ = remove_self_loops(edge_index2)
                    edge_index2 = torch.cat([edge_index, edge_index2], dim=1)
                else:
                    edge_index2 = edge_index
                
                m_idx = (y_hard[..., 0] == 1).nonzero().unique()

                g = self.pool_edge(edge_index2, m_idx, num_nodes=num_nodes)
                g, _ = coalesce(g, None, num_nodes=len(m_idx))

                m_gs.append(g)
                m_ids.append(m_idx)
                assert len(m_ids) == i + 1
            # record the info
            down_outs.append(h)
            down_ps.append(pos)
            # inter-level fusion
            tmp_g = gs_main if self.esn > 1 else gs
            h = self.edge_conv(h, tmp_g, ew)
            if self.agg_conv_pos:
                pos = self.edge_conv(pos, tmp_g, ew)
            cts.append(ew)
            # pooling
            h = self._pool_tensor(h, m_ids, y_hard, i, i < self.pre_l_n)
            pos = self._pool_tensor(pos, m_ids, y_hard, i, i < self.pre_l_n)
        
        for l in range(self.bottom_ln):
            h, ew = self.bottom_gmp[l](h, m_gs[l_n], pos)
            if self.lagrangian and l == 0:
                h, ew = self.bottom_gmp[l](h, m_gs[l_n], pos)
        
        # up pass
        for i in range(l_n):
            up_idx = l_n - i - 1
            g, idx = m_gs[up_idx], m_ids[up_idx]
            if self.esn > 1:
                g_main, _ = add_remaining_self_loops(g[0])
                g_cont, _ = add_remaining_self_loops(g[1]) 
                g = [g_main, g_cont]
            else:
                g, _ = add_remaining_self_loops(g)
            h = self.unpools[i](h, down_outs[up_idx].shape[-2], idx)
            tmp_g = g[0] if self.esn > 1 else g
            h= self.edge_conv(h, tmp_g, cts[up_idx], aggragating=False)
            h, ew_u = self.up_gmps[i](h, g, down_ps[up_idx])
            if up_idx == 0 and self.lagrangian:
                h, ew_u = self.up_gmps[i](h, g, down_ps[up_idx])
            h = h + down_outs[up_idx]
        return h

