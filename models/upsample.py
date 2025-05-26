import torch
from torch import nn
from utils import ops
import math
import torch.nn.functional as F

class UpSample(nn.Module):
    def __init__(self, config_upsample, layer):
        super(UpSample, self).__init__()
        q_in       = config_upsample.q_in[layer] 
        q_out      = config_upsample.q_out[layer] 
        k_in       = config_upsample.k_in[layer] 
        k_out      = config_upsample.k_out[layer]  
        v_in       = config_upsample.v_in[layer] 
        v_out      = config_upsample.v_out[layer] 
        num_heads  = config_upsample.num_heads[layer] 

        self.num_heads = num_heads
        self.q_depth = int(q_out / num_heads)
        self.k_depth = int(k_out / num_heads)
        self.v_depth = int(v_out / num_heads)

        self.q_conv = nn.Conv1d(q_in, q_out, 1, bias=False)
        self.k_conv = nn.Conv1d(k_in, k_out, 1, bias=False)
        self.v_conv = nn.Conv1d(v_in, v_out, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.skip_link = nn.Conv1d(q_in, v_out, 1, bias=False)

    def forward(self, pcd_up, pcd_down, pcd_up_xyz):
        (points_select, idx_select, points_select_xyz), (points_drop, idx_drop) = pcd_down
        # pcd_up.shape == (B, C, M1)  points_select.shape == (B, C, M2)
        q = self.q_conv(pcd_up)
        # q.shape == (B, C, M1)
        q = self.split_heads(q, self.num_heads, self.q_depth)
        # q.shape == (B, H, D, M1)
        k = self.k_conv(points_select)
        # k.shape == (B, C, M2)
        k = self.split_heads(k, self.num_heads, self.k_depth)
        # k.shape == (B, H, D, M2)
        v = self.v_conv(points_select)
        # v.shape == (B, C, M2)
        v = self.split_heads(v, self.num_heads, self.v_depth)
        # v.shape == (B, H, D, M2)
        energy = q.permute(0, 1, 3, 2) @ k
        # energy.shape == (B, H, M1, M2)
        scale_factor = math.sqrt(q.shape[-2])
        attention = self.softmax(energy / scale_factor)
        # attention.shape == (B, H, M1, M2)
        x = (attention @ v.permute(0, 1, 3, 2)).permute(0, 2, 1, 3)
        # x.shape == (B, M1, H, D)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        # x.shape == (B, C, M1)
        x = self.skip_link(pcd_up) + x
        return x

    def split_heads(self, x, heads, depth):
        # x.shape == (B, C, N)
        x = x.view(x.shape[0], heads, depth, x.shape[2])
        # x.shape == (B, H, D, N)
        return x


class UpSampleSelfAttention(nn.Module):
    def __init__(self, config_upsample, layer):
        super(UpSampleSelfAttention, self).__init__()
        q_in       = config_upsample.q_in[layer] 
        q_out      = config_upsample.q_out[layer] 
        k_in       = config_upsample.k_in[layer] 
        k_out      = config_upsample.k_out[layer]  
        v_in       = config_upsample.v_in[layer] 
        v_out      = config_upsample.v_out[layer] 
        num_heads  = config_upsample.num_heads[layer] 

        self.num_heads = num_heads
        self.q_depth = int(q_out / num_heads)
        self.k_depth = int(k_out / num_heads)
        self.v_depth = int(v_out / num_heads)

        self.q_conv = nn.Conv1d(q_in, q_out, 1, bias=False)
        self.k_conv = nn.Conv1d(k_in, k_out, 1, bias=False)
        self.v_conv = nn.Conv1d(v_in, v_out, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.skip_link = nn.Conv1d(q_in, v_out, 1, bias=False)

    def forward(self, pcd_up, pcd_down, pcd_up_xyz):
        (points_select, idx_select, points_select_xyz), (points_drop, idx_drop) = pcd_down
        # points_select.shape == (B, C, M1)  points_drop.shape == (B, C, M2)
        x = self.concat_by_idx(points_select, points_drop, idx_select, idx_drop, dim=-1)
        # x.shape == (B, C, N)
        q = self.q_conv(x)
        # q.shape == (B, C, N)
        q = self.split_heads(q, self.num_heads, self.q_depth)
        # q.shape == (B, H, D, N)
        k = self.k_conv(x)
        # k.shape == (B, C, N)
        k = self.split_heads(k, self.num_heads, self.k_depth)
        # k.shape == (B, H, D, N)
        v = self.v_conv(x)
        # v.shape == (B, C, N)
        v = self.split_heads(v, self.num_heads, self.v_depth)
        # v.shape == (B, H, D, N)
        energy = q.permute(0, 1, 3, 2) @ k
        # energy.shape == (B, H, N, N)
        scale_factor = math.sqrt(q.shape[-2])
        attention = self.softmax(energy / scale_factor)
        # attention.shape == (B, H, N, N)
        x = (attention @ v.permute(0, 1, 3, 2)).permute(0, 2, 1, 3)
        # x.shape == (B, N, H, D)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        # x.shape == (B, C, N)
        x = self.skip_link(pcd_up) + x
        return x

    def split_heads(self, x, heads, depth):
        # x.shape == (B, C, N)
        x = x.view(x.shape[0], heads, depth, x.shape[2])
        # x.shape == (B, H, D, N)
        return x

    def concat_by_idx(self, a, b, idx_a, idx_b, dim):
        src = torch.cat([a, b], dim=dim)
        target = torch.zeros_like(src)
        idx_tmp = torch.cat([idx_a, idx_b], dim=dim).expand(-1, a.shape[1], -1)
        return target.scatter(dim=-1, index=idx_tmp, src=src)


class UpSampleInterpolation(nn.Module):
    def __init__(self, config_upsample, layer):
        super(UpSampleInterpolation, self).__init__()
        q_in       = config_upsample.q_in[layer] 
        q_out      = config_upsample.q_out[layer] 
        k_in       = config_upsample.k_in[layer] 
        k_out      = config_upsample.k_out[layer]  
        v_in       = config_upsample.v_in[layer] 
        v_out      = config_upsample.v_out[layer] 
        num_heads  = config_upsample.num_heads[layer]
        
        self.distance_type  = config_upsample.interpolation.distance_type[layer]
        self.K              = config_upsample.interpolation.K[layer] 

        self.conv = nn.Sequential(nn.Conv1d(q_in, v_out, 1, bias=False), nn.BatchNorm1d(v_out), nn.LeakyReLU(negative_slope=0.2))
        # self.skip_link = nn.Sequential(nn.Conv1d(q_in, v_out, 1, bias=False), nn.BatchNorm1d(v_out), nn.LeakyReLU(negative_slope=0.2))
        self.res_conv = nn.Sequential(nn.Conv1d(2*v_out, v_out, 1, bias=False), nn.BatchNorm1d(v_out), nn.LeakyReLU(negative_slope=0.2))
    def forward(self, pcd_up, pcd_down, pcd_up_xyz):
        (points_select, idx_select, points_select_xyz), (points_drop, idx_drop) = pcd_down
        # pcd_up.shape == (B, C, N)  points_select.shape == (B, C, M)
        interpolated_points = self.interpolate(pcd_up, points_select, pcd_up_xyz, points_select_xyz, distance_type=self.distance_type, K=self.K)
        # x = self.skip_link(pcd_up) + interpolated_points
        x = torch.concat([pcd_up, interpolated_points], dim=1)
        x = self.res_conv(x)
        return x
    
    def interpolate(self, pcd_up, points_select, pcd_up_xyz, points_select_xyz, distance_type="feature", K=3):
        points_select_conv = self.conv(points_select) # points_select_conv.shape == (B, C, M)
        if distance_type == "feature":
            neighbors, _, d_neighbors = ops.select_neighbors_interpolate(pcd_up, points_select, points_select_conv, K=K)
        elif distance_type == "xyz":
            neighbors, _, d_neighbors = ops.select_neighbors_interpolate(pcd_up_xyz, points_select_xyz, points_select_conv, K=K)
        else:
            raise ValueError(f'upsample interpolation distance type can only be feature or xyz! Got: {distance_type}')
        # neighbors.shape == (B, C, N, K), idx.shape == (B, N, K)
        weights = 1.0 / (d_neighbors + 1e-8)
        weights = weights / torch.sum(weights, dim=-1, keepdim=True) # weights.shape == (B, N, K)
        interpolated_points = torch.sum(neighbors * weights.unsqueeze(dim=1), dim=-1) # interpolated_points.shape == (B, C, N), dim=-1)
        return interpolated_points
    
        

    
