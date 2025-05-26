import torch
from torch import nn
from utils import ops
import torch.nn.init as init

class EdgeConv(nn.Module):
    def __init__(self, config_embedding, layer):

        super(EdgeConv, self).__init__()
        self.K              = config_embedding.K[layer]
        self.group_type     = config_embedding.group_type[layer]
        self.normal_channel = config_embedding.normal_channel
        conv1_channel_in    = config_embedding.conv1_in[layer]
        conv1_channel_out   = config_embedding.conv1_out[layer]
        conv2_channel_in    = config_embedding.conv2_in[layer]
        conv2_channel_out   = config_embedding.conv2_out[layer]

        self.conv1 = nn.Sequential(nn.Conv2d(conv1_channel_in, conv1_channel_out, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(conv1_channel_out),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(conv2_channel_in, conv2_channel_out, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(conv2_channel_out),
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        # x.shape == (B, C, N)
        x, _ = ops.group(x, self.K, self.group_type, self.normal_channel)
        # x.shape == (B, 2C, N, K) or (B, C, N, K)
        x = self.conv1(x)
        # x.shape == (B, C, N, K)
        x = self.conv2(x)
        # x.shape == (B, C, N, K)
        x = x.max(dim=-1, keepdim=False)[0]
        # x.shape == (B, C, N)
        return x

class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        # self.args = args
        # self.k = 3
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Sequential(nn.Linear(1024, 512, bias=False),
                                     nn.BatchNorm1d(512),
                                     nn.LeakyReLU(negative_slope=0.2))
        self.linear2 = nn.Sequential(nn.Linear(512, 256, bias=False),
                                     nn.BatchNorm1d(256),
                                     nn.LeakyReLU(negative_slope=0.2))
        self.transform = nn.Linear(256, 3 * 3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))
        self.dp1 = nn.Dropout(p=0.5)
        self.dp2 = nn.Dropout(p=0.5)

    def forward(self, x):
        B = x.size(0)

        x = self.conv1(x)  # (B, 3*2, num_points, k) -> (B, 64, num_points, k)
        x = self.conv2(x)  # (B, 64, num_points, k) -> (B, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]  # (B, 128, num_points, k) -> (B, 128, num_points)

        x = self.conv3(x)  # (B, 128, num_points) -> (B, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]  # (B, 1024, num_points) -> (B, 1024)

        x = self.linear1(x)  # (B, 1024) -> (B, 512)
        x = self.dp1(x)
        x = self.linear2(x)  # (B, 512) -> (B, 256)
        x = self.dp2(x)
        x = self.transform(x)  # (B, 256) -> (B, 3*3)
        x = x.view(B, 3, 3)  # (B, 3*3) -> (B, 3, 3)

        return x
