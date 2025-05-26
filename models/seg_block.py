import torch
from torch import nn
from utils import ops
from models import embedding
from models import attention
from models import downsample
from models import upsample


class FeatureLearningBlock(nn.Module):
    def __init__(self, config_feature_learning_block):
        super(FeatureLearningBlock, self).__init__()
        downsample_which = config_feature_learning_block.downsample.ds_which
        upsample_which = config_feature_learning_block.upsample.us_which

        self.embedding_list = nn.ModuleList(
            [
                embedding.EdgeConv(config_feature_learning_block.embedding, layer)
                for layer in range(len(config_feature_learning_block.embedding.K))
            ]
        )
        if downsample_which == "global":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSample(
                        config_feature_learning_block.downsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.downsample.M))
                ]
            )
        elif downsample_which == "local":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSampleWithSigma(
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
        elif downsample_which == "local_insert":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSampleInsert(
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
            raise ValueError(
                "Only global_carve and local_insert are valid for ds_which!"
            )
        self.feature_learning_layer_list = nn.ModuleList(
            [
                attention.Neighbor2PointAttention(
                    config_feature_learning_block.attention, layer
                )
                for layer in range(len(config_feature_learning_block.attention.K))
            ]
        )

        if upsample_which == "crossA":
            self.upsample_list = nn.ModuleList(
                [
                    upsample.UpSample(config_feature_learning_block.upsample, layer)
                    for layer in range(len(config_feature_learning_block.upsample.q_in))
                ]
            )
        elif upsample_which == "selfA":
            self.upsample_list = nn.ModuleList(
                [
                    upsample.UpSampleSelfAttention(
                        config_feature_learning_block.upsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.upsample.q_in))
                ]
            )
        elif upsample_which == "interpolation":
            self.upsample_list = nn.ModuleList(
                [
                    upsample.UpSampleInterpolation(
                        config_feature_learning_block.upsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.upsample.q_in))
                ]
            )
        else:
            raise ValueError("Only crossA and selfA are valid for us_which!")

    def forward(self, x):
        x_xyz = x[:, :3, :]
        x_list = []
        for embedding in self.embedding_list:
            x = embedding(x)
            x_list.append(x)
        x = torch.cat(x_list, dim=1)
        x = self.feature_learning_layer_list[0](x)
        x_list = [x]
        points_drop_list = []
        idx_select_list = []
        idx_drop_list = []
        x_xyz_list = [x_xyz]
        for i in range(len(self.downsample_list)):
            (x, idx_select), (points_drop, idx_drop) = self.downsample_list[i](x, x_xyz)
            x = self.feature_learning_layer_list[i + 1](x)
            x_xyz = ops.gather_by_idx(x_xyz, idx_select)
            x_list.append(x)
            x_xyz_list.append(x_xyz)
            points_drop_list.append(points_drop)
            idx_select_list.append(idx_select)
            idx_drop_list.append(idx_drop)
        split = int((len(self.feature_learning_layer_list) - 1) / 2)
        x = (
            (x_list.pop(), idx_select_list.pop(), x_xyz_list.pop()),
            (points_drop_list.pop(), idx_drop_list.pop()),
        )
        for j in range(len(self.upsample_list)):
            x_tmp = x_list.pop()
            x_xyz_tmp = x_xyz_list[-1 - j]
            x = self.upsample_list[j](x_tmp, x, x_xyz_tmp)
            x = self.feature_learning_layer_list[j + 1 + split](x)
            if j < len(self.upsample_list) - 1:
                x = (
                    (x, idx_select_list.pop(), x_xyz_list[-1 - j]),
                    (points_drop_list.pop(), idx_drop_list.pop()),
                )
        return x
