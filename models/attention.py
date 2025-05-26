from torch import nn
from utils import ops
import math
import torch

class L2Attention(nn.Module):
    def __init__(self, config_attention, layer):
        super(Neighbor2PointAttention, self).__init__()
        self.K = config_attention.K[layer]
        self.group_type = config_attention.group_type[layer]
        self.num_heads = config_attention.num_heads[layer]
        self.attention_mode = config_attention.attention_mode[layer]
        q_in = config_attention.q_in[layer]
        q_out = config_attention.q_out[layer]
        k_in = config_attention.k_in[layer]
        k_out = config_attention.k_out[layer]
        v_in = config_attention.v_in[layer]
        v_out = config_attention.v_out[layer]
        ff_conv1_channels_in = config_attention.ff_conv1_channels_in[layer]
        ff_conv1_channels_out = config_attention.ff_conv1_channels_out[layer]
        ff_conv2_channels_in = config_attention.ff_conv2_channels_in[layer]
        ff_conv2_channels_out = config_attention.ff_conv2_channels_out[layer]
        self.asm = config_attention.asm[layer]

        self.q_depth = int(q_out / self.num_heads)
        self.k_depth = int(k_out / self.num_heads)
        self.v_depth = int(v_out / self.num_heads)

        self.q_conv = nn.Conv2d(q_in, q_out, 1, bias=False)
        self.k_conv = nn.Conv2d(k_in, k_out, 1, bias=False)
        self.v_conv = nn.Conv2d(v_in, v_out, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.ff = nn.Sequential(nn.Conv1d(ff_conv1_channels_in, ff_conv1_channels_out, 1, bias=False),
                                nn.LeakyReLU(negative_slope=0.2),
                                nn.Conv1d(ff_conv2_channels_in, ff_conv2_channels_out, 1, bias=False))
        self.bn1 = nn.BatchNorm1d(v_out)
        self.bn2 = nn.BatchNorm1d(v_out)

    def forward(self, x):
        # x.shape == (B, C, N)
        neighbors, _ = ops.group(x, self.K, self.group_type)
        # neighbors.shape == (B, C, N, K)
        x_tmp = x[:, :, :, None]
        # x_tmp.shape == (B, C, N, 1)
        q = self.q_conv(x_tmp)
        # q.shape == (B, C, N, 1)
        q = self.split_heads(q, self.num_heads, self.q_depth)
        # q.shape == (B, H, N, 1, D)
        k = self.k_conv(neighbors)
        # k.shape ==  (B, C, N, K)
        k = self.split_heads(k, self.num_heads, self.k_depth)
        # k.shape == (B, H, N, K, D)
        v = self.v_conv(neighbors)
        # v.shape ==  (B, C, N, K)
        v = self.split_heads(v, self.num_heads, self.v_depth)
        # v.shape == (B, H, N, K, D)
        k = k.permute(0, 1, 2, 4, 3)
        # k.shape == (B, H, N, D, K)
        x_tmp = self.attention(q, k, v)
        # x_tmp.shape == (B, C, N)
        x = self.bn1(x + x_tmp)
        # x.shape == (B, C, N)
        x_tmp = self.ff(x)
        # x_tmp.shape == (B, C, N)
        x = self.bn2(x + x_tmp)
        # x.shape == (B, C, N)
        return x

    def split_heads(self, x, heads, depth):
        # x.shape == (B, C, N, K)
        x = x.view(x.shape[0], heads, depth, x.shape[2], x.shape[3])
        # x.shape == (B, H, D, N, K)
        x = x.permute(0, 1, 3, 4, 2)
        # x.shape == (B, H, N, K, D)
        return x

    def attention(self, q, k, v):
        """
        Local based attention variation input shape:
        q.shape == (B, H, N, 1, D)
        k.shape == (B, H, N, D, K)
        v.shape == (B, H, N, K, D)
        """
        if self.attention_mode == "scalar_dot":
            attention = self.attention_scoring(q, k)  # attention.shape == (B, H, N, 1, K)
            x = (attention @ v)[:, :, :, 0, :].permute(0, 2, 1, 3)
            # x.shape == (B, N, H, D)
        elif self.attention_mode == "vector_sub":
            energy = q.repeat(1, 1, 1, k.shape[-1], 1) - k.permute(0, 1, 2, 4, 3)
            # energy.shape == (B, H, N, K, D)
            scale_factor = math.sqrt(q.shape[-1])
            attention = self.softmax(energy / scale_factor)
            # attention.shape == (B, H, N, K, D)
            x = (attention * v).permute(0, 2, 1, 3, 4)
            # x.shape == (B, N, H, K, D) element-wise multiplication
            x = x.sum(dim=-2)
            # x.shape == (B, N, H, D)
        else:
            raise ValueError('attention_mode can only be scalar_dot or vector_sub, but got: {self.attention_mode}')
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # x_tmp.shape == (B, C, N)
        return x

    def attention_scoring(self, q, k):  # q.shape == B, H, N, 1, D), k.shape == (B, H, N, D, K)
        if self.asm == "dot":
            energy = q @ k  # energy.shape == (B, H, N, 1, K)
        elif self.asm == "dot-sub":
            energy = q @ (q.transpose(-1, -2) - k)  # Q@(Q-K)
        else:
            raise ValueError('Please check the setting of asm in feature learning layer!')
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)  # attention.shape == (B, H, N, 1, K)
        return attention

class Neighbor2PointAttention(nn.Module):
    def __init__(self, config_attention, layer):
        super(Neighbor2PointAttention, self).__init__()
        self.K = config_attention.K[layer]
        self.group_type = config_attention.group_type[layer]
        self.num_heads = config_attention.num_heads[layer]
        self.attention_mode = config_attention.attention_mode[layer]
        q_in = config_attention.q_in[layer]
        q_out = config_attention.q_out[layer]
        k_in = config_attention.k_in[layer]
        k_out = config_attention.k_out[layer]
        v_in = config_attention.v_in[layer]
        v_out = config_attention.v_out[layer]
        ff_conv1_channels_in = config_attention.ff_conv1_channels_in[layer]
        ff_conv1_channels_out = config_attention.ff_conv1_channels_out[layer]
        ff_conv2_channels_in = config_attention.ff_conv2_channels_in[layer]
        ff_conv2_channels_out = config_attention.ff_conv2_channels_out[layer]
        self.asm = config_attention.asm[layer]

        self.q_depth = int(q_out / self.num_heads)
        self.k_depth = int(k_out / self.num_heads)
        self.v_depth = int(v_out / self.num_heads)

        self.q_conv = nn.Conv2d(q_in, q_out, 1, bias=False)
        self.k_conv = nn.Conv2d(k_in, k_out, 1, bias=False)
        self.v_conv = nn.Conv2d(v_in, v_out, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.ff = nn.Sequential(nn.Conv1d(ff_conv1_channels_in, ff_conv1_channels_out, 1, bias=False),
                                nn.LeakyReLU(negative_slope=0.2),
                                nn.Conv1d(ff_conv2_channels_in, ff_conv2_channels_out, 1, bias=False))
        self.bn1 = nn.BatchNorm1d(v_out)
        self.bn2 = nn.BatchNorm1d(v_out)

    def forward(self, x):
        # x.shape == (B, C, N)
        neighbors, _ = ops.group(x, self.K, self.group_type)
        # neighbors.shape == (B, C, N, K)
        x_tmp = x[:, :, :, None]
        # x_tmp.shape == (B, C, N, 1)
        q = self.q_conv(x_tmp)
        # q.shape == (B, C, N, 1)
        q = self.split_heads(q, self.num_heads, self.q_depth)
        # q.shape == (B, H, N, 1, D)
        k = self.k_conv(neighbors)
        # k.shape ==  (B, C, N, K)
        k = self.split_heads(k, self.num_heads, self.k_depth)
        # k.shape == (B, H, N, K, D)
        v = self.v_conv(neighbors)
        # v.shape ==  (B, C, N, K)
        v = self.split_heads(v, self.num_heads, self.v_depth)
        # v.shape == (B, H, N, K, D)
        k = k.permute(0, 1, 2, 4, 3)
        # k.shape == (B, H, N, D, K)
        x_tmp = self.attention(q, k, v)
        # x_tmp.shape == (B, C, N)
        x = self.bn1(x + x_tmp)
        # x.shape == (B, C, N)
        x_tmp = self.ff(x)
        # x_tmp.shape == (B, C, N)
        x = self.bn2(x + x_tmp)
        # x.shape == (B, C, N)
        return x

    def split_heads(self, x, heads, depth):
        # x.shape == (B, C, N, K)
        x = x.view(x.shape[0], heads, depth, x.shape[2], x.shape[3])
        # x.shape == (B, H, D, N, K)
        x = x.permute(0, 1, 3, 4, 2)
        # x.shape == (B, H, N, K, D)
        return x

    def attention(self, q, k, v):
        """
        Local based attention variation input shape:
        q.shape == (B, H, N, 1, D)
        k.shape == (B, H, N, D, K)
        v.shape == (B, H, N, K, D)
        """
        if self.attention_mode == "scalar_dot":
            attention = self.attention_scoring(q, k)  # attention.shape == (B, H, N, 1, K)
            x = (attention @ v)[:, :, :, 0, :].permute(0, 2, 1, 3)
            # x.shape == (B, N, H, D)
        elif self.attention_mode == "vector_sub":
            energy = q.repeat(1, 1, 1, k.shape[-1], 1) - k.permute(0, 1, 2, 4, 3)
            # energy.shape == (B, H, N, K, D)
            scale_factor = math.sqrt(q.shape[-1])
            attention = self.softmax(energy / scale_factor)
            # attention.shape == (B, H, N, K, D)
            x = (attention * v).permute(0, 2, 1, 3, 4)
            # x.shape == (B, N, H, K, D) element-wise multiplication
            x = x.sum(dim=-2)
            # x.shape == (B, N, H, D)
        else:
            raise ValueError('attention_mode can only be scalar_dot or vector_sub, but got: {self.attention_mode}')
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # x_tmp.shape == (B, C, N)
        return x

    def attention_scoring(self, q, k):  # q.shape == B, H, N, 1, D), k.shape == (B, H, N, D, K)
        if self.asm == "dot":
            energy = q @ k  # energy.shape == (B, H, N, 1, K)
        elif self.asm == "dot-sub":
            energy = q @ (q.transpose(-1, -2) - k)  # Q@(Q-K)
        else:
            raise ValueError('Please check the setting of asm in feature learning layer!')
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)  # attention.shape == (B, H, N, 1, K)
        return attention


class Point2PointAttention(nn.Module):
    def __init__(self, config_attention, layer):
        num_heads = config_attention.num_heads[layer]
        self.attention_mode = config_attention.attention_mode[layer]
        q_in = config_attention.q_in[layer]
        q_out = config_attention.q_out[layer]
        k_in = config_attention.k_in[layer]
        k_out = config_attention.k_out[layer]
        v_in = config_attention.v_in[layer]
        v_out = config_attention.v_out[layer]
        ff_conv1_channels_in = config_attention.ff_conv1_channels_in[layer]
        ff_conv1_channels_out = config_attention.ff_conv1_channels_out[layer]
        ff_conv2_channels_in = config_attention.ff_conv2_channels_in[layer]
        ff_conv2_channels_out = config_attention.ff_conv2_channels_out[layer]
        self.asm = config_attention.asm[layer]

        super(Point2PointAttention, self).__init__()
        # check input values
        if q_in != k_in or q_in != v_in or k_in != v_in:
            raise ValueError(f'q_in, k_in and v_in should be the same! Got q_in:{q_in}, k_in:{k_in}, v_in:{v_in}')
        if q_out != k_out:
            raise ValueError('q_out should be equal to k_out!')
        if q_out % num_heads != 0 or k_out % num_heads != 0 or v_out % num_heads != 0:
            raise ValueError('please set another value for num_heads!')
        if q_in != v_out:
            raise ValueError(f'q_in should be equal to v_out due to ResLink! Got q_in: {q_in}, v_out: {v_out}')

        self.num_heads = num_heads
        self.q_depth = int(q_out / num_heads)
        self.k_depth = int(k_out / num_heads)
        self.v_depth = int(v_out / num_heads)

        self.q_conv = nn.Conv1d(q_in, q_out, 1, bias=False)
        self.k_conv = nn.Conv1d(k_in, k_out, 1, bias=False)
        self.v_conv = nn.Conv1d(v_in, v_out, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.ff = nn.Sequential(nn.Conv1d(ff_conv1_channels_in, ff_conv1_channels_out, 1, bias=False),
                                nn.LeakyReLU(negative_slope=0.2),
                                nn.Conv1d(ff_conv2_channels_in, ff_conv2_channels_out, 1, bias=False))
        self.bn1 = nn.BatchNorm1d(v_out)
        self.bn2 = nn.BatchNorm1d(v_out)

    def forward(self, x):
        # x.shape == (B, C, N)
        q = self.q_conv(x)
        # q.shape == (B, C, N)
        q = self.split_heads(q, self.num_heads, self.q_depth)
        # q.shape == (B, H, D, N)
        k = self.k_conv(x)
        # k.shape ==  (B, C, N)
        k = self.split_heads(k, self.num_heads, self.k_depth)
        # k.shape == (B, H, D, N)
        v = self.v_conv(x)
        # v.shape ==  (B, C, N)
        v = self.split_heads(v, self.num_heads, self.v_depth)
        # v.shape == (B, H, D, N)

        q = q.permute(0, 1, 3, 2)
        attention = self.attention_scoring(q, k)  # attention.shape == (B, H, N, N)

        x_tmp = (attention @ v.permute(0, 1, 3, 2)).permute(0, 2, 1, 3)
        # x_tmp.shape == (B, N, H, D)
        x_tmp = x_tmp.reshape(x_tmp.shape[0], x_tmp.shape[1], -1).permute(0, 2, 1)
        # x_tmp.shape == (B, C, N)
        x = self.bn1(x + x_tmp)
        # x.shape == (B, C, N)
        x_tmp = self.ff(x)
        # x_tmp.shape == (B, C, N)
        x = self.bn2(x + x_tmp)
        # x.shape == (B, C, N)
        return x

    def split_heads(self, x, heads, depth):
        # x.shape == (B, C, N)
        x = x.view(x.shape[0], heads, depth, x.shape[2])
        # x.shape == (B, H, D, N)
        return x

    def attention_scoring(self, q, k):  # q.shape == (B, H, N, D), k.shape == (B, H, D, N)
        if self.asm == "dot":
            energy = q @ k  # energy.shape == (B, H, N, N)
        elif self.asm == "l2":
            energy = -1 * ops.l2_global(q, k)  # -(Q-K)^2 energy.shape == (B, H, N, N)
        elif self.asm == "l2+":
            energy = ops.l2_global(q, k)  # (Q-K)^2 energy.shape == (B, H, N, N)
        else:
            raise ValueError('Please check the setting of asm in feature learning layer!')
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)  # attention.shape == (B, H, N, N)
        return attention
