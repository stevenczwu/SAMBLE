import torch
from torch import nn
import time

from models import attention, downsample, embedding
from utils.ops import farthest_point_sample
from utils import ops


class FeatureLearningBlock(nn.Module):
    def __init__(self, config_feature_learning_block, fps=False):
        downsample_which = config_feature_learning_block.downsample.ds_which
        ff_conv2_channels_out = (
            config_feature_learning_block.attention.ff_conv2_channels_out
        )
        self.res_link_enable = config_feature_learning_block.res_link.enable
        fl_which = config_feature_learning_block.attention.fl_which

        super(FeatureLearningBlock, self).__init__()
        self.embedding_list = nn.ModuleList(
            [
                embedding.EdgeConv(config_feature_learning_block.embedding, layer)
                for layer in range(len(config_feature_learning_block.embedding.K))
            ]
        )
        if downsample_which == "global":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSampleGlobal(
                        config_feature_learning_block.downsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.downsample.M))
                ]
            )
        elif downsample_which == "global_carve":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSampleCarve(
                        config_feature_learning_block.downsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.downsample.M))
                ]
            )
        elif downsample_which == "local":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSampleLocal(
                        config_feature_learning_block.downsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.downsample.M))
                ]
            )
        elif downsample_which == "token":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSampleToken(
                        config_feature_learning_block.downsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.downsample.M))
                ]
            )
        else:
            raise NotImplementedError
        if fl_which == "n2p":
            self.feature_learning_layer_list = nn.ModuleList(
                [
                    attention.Neighbor2PointAttention(
                        config_feature_learning_block.attention, layer
                    )
                    for layer in range(len(config_feature_learning_block.attention.K))
                ]
            )
        elif fl_which == "p2p":
            self.feature_learning_layer_list = nn.ModuleList(
                [
                    attention.Point2PointAttention(
                        config_feature_learning_block.attention, layer
                    )
                    for layer in range(len(config_feature_learning_block.attention.K))
                ]
            )
        else:
            raise ValueError("Only n2p and p2p are valid for fl_which!")

        if self.res_link_enable:
            self.conv_list = nn.ModuleList(
                [
                    nn.Conv1d(channel_in, 1024, kernel_size=1, bias=False)
                    for channel_in in ff_conv2_channels_out
                ]
            )

        else:
            self.conv = nn.Conv1d(
                ff_conv2_channels_out[-1], 1024, kernel_size=1, bias=False
            )

        self.M_list = config_feature_learning_block.downsample.M

        self.fps = fps

    def forward(self, x):
        x_list = []
        x_xyz = x.clone()
        for embedding in self.embedding_list:
            x = embedding(x)
            x_list.append(x)
        x = torch.cat(x_list, dim=1)
        x = self.feature_learning_layer_list[0](x)

        if self.res_link_enable:
            res_link_list = []
            res_link_list.append(self.conv_list[0](x).max(dim=-1)[0])
            for i in range(len(self.downsample_list)):
                # x_xyz.shape, torch.Size([8, 3, 2048])
                # x.shape, torch.Size([8, 128, 2048])
                if self.fps:
                    x_idx = farthest_point_sample(
                        torch.permute(x_xyz, (0, 2, 1)), self.M_list[i] * 2
                    )  # [B, npoint]
                    x = torch.gather(
                        x, 2, x_idx.unsqueeze(1).expand(-1, x.shape[1], -1)
                    )
                    x_xyz_down = torch.gather(
                        x_xyz, 2, x_idx.unsqueeze(1).expand(-1, 3, -1)
                    )

                    (x, idx_select) = self.downsample_list[i](x, x_xyz_down)[0]

                    idx_select = torch.gather(x_idx.unsqueeze(1), 2, idx_select)
                else:
                    (x, idx_select) = self.downsample_list[i](x, x_xyz)[0]

                x = self.feature_learning_layer_list[i + 1](x)
                x_xyz = ops.gather_by_idx(x_xyz, idx_select)
                res_link_list.append(self.conv_list[i + 1](x).max(dim=-1)[0])
            self.res_link_list = res_link_list
            x = torch.cat(res_link_list, dim=1)
            return x, res_link_list
        else:
            for i in range(len(self.downsample_list)):
                x = self.downsample_list[i](x)[0][0]
                x = self.feature_learning_layer_list[i + 1](x)
            x = self.conv(x).max(dim=-1)[0]
            return x


class ModelNetModel(nn.Module):
    def __init__(self, config, fps=False):
        super(ModelNetModel, self).__init__()

        if config.feature_learning_block.enable:
            self.block = FeatureLearningBlock(config.feature_learning_block, fps)
            num_layers = len(config.feature_learning_block.attention.K)
        else:
            raise ValueError("Only neighbor2point block supported!")

        num_output = 40

        self.res_link_enable = config.feature_learning_block.res_link.enable

        if self.res_link_enable:
            self.linear1 = nn.Sequential(
                nn.Linear(1024 * num_layers, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(p=0.5),
            )
            self.linear2 = nn.Sequential(
                nn.Linear(1024, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(p=0.5),
            )
            self.linear3 = nn.Linear(256, num_output)
        else:
            self.linear2 = nn.Sequential(
                nn.Linear(1024, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(p=0.5),
            )
            self.linear3 = nn.Linear(256, num_output)

    def forward(self, x):  # x.shape == (B, 3, N)
        if self.res_link_enable:
            # with res_link
            x, x_res_link_list = self.block(x)  # x.shape == (B, 3C)

            # no_aux
            x = self.MLP(x)  # x.shape == (B, 40)

            return x
        else:
            # no_res_link
            x = self.block(x)  # x.shape == (B, 1024)
            x = self.MLP_no_res(x)  # x.shape == (B, 40)
            return x

    def MLP(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

    def MLP_no_res(self, x):
        x = self.linear2(x)
        x = self.linear3(x)
        return x

    def MLP_unshared(self, x, i):
        x = self.linear1_list[i](x)  # B * 1024
        x = self.linear2_list[i](x)  # B * 512
        x = self.linear3_list[i](x)  # B * 40
        return x

    def MLP_concat(self, x):
        x = self.linear0(x)
        x = self.MLP(x)
        return x

    def MLP_unshared_concat(self, x, i):
        x = self.linear0(x)
        x = self.MLP_unshared(x, i)
        return x
