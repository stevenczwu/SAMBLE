import torch
from torch import nn
from models import attention, downsample, embedding, upsample
from utils import ops


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


class ShapeNetModel(nn.Module):
    def __init__(self, config):
        super(ShapeNetModel, self).__init__()

        if config.feature_learning_block.enable:
            self.block = FeatureLearningBlock(config.feature_learning_block)
            output_channels = (
                config.feature_learning_block.attention.ff_conv2_channels_out[-1]
            )
        self.conv = nn.Sequential(
            nn.Conv1d(output_channels, 1024, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(output_channels + 2048 + 64, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(1024, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Conv1d(256, 50, kernel_size=1, bias=False)
        self.dp1 = nn.Dropout(p=0.5)
        self.dp2 = nn.Dropout(p=0.5)
        self.STN_enable = config.feature_learning_block.STN
        if self.STN_enable == True:
            self.STN = embedding.STN()

        self.stn_regularization_loss_factor = (
            config.train.stn_regularization_loss_factor
        )

    def forward(self, x, category_id):
        # x.shape == (B, 3, N)  category_id.shape == (B, 16, 1)
        B, C, N = x.shape
        # x.shape == (B, 3, N)

        if self.STN_enable:
            x0, _ = ops.group(
                x, 32, "center_diff"
            )  # (B, 3, num_points) -> (B, 3*2, num_points, k)
            trans = self.STN(x0)  # (B, 3, 3)
            x = x.transpose(2, 1)  # (B, 3, num_points) -> (B, num_points, 3)
            x = torch.bmm(
                x, trans
            )  # (B, num_points, 3) * (B, 3, 3) -> (B, num_points, 3)
            x = x.transpose(2, 1)  # (B, num_points, 3) -> (B, 3, num_points)

        x_tmp = self.block(x)
        # x_tmp.shape == (B, C, N)
        x = self.conv(x_tmp)
        # x.shape == (B, 1024, N)
        x_max = x.max(dim=-1, keepdim=True)[0]
        # x_max.shape == (B, 1024, 1)
        x_average = x.mean(dim=-1, keepdim=True)
        # x_average.shape == (B, 1024, 1)
        x = torch.cat([x_max, x_average], dim=1)
        # x.shape == (B, 2048, 1)
        category_id = self.conv1(category_id)
        # category_id.shape == (B, 64, 1)
        x = torch.cat([x, category_id], dim=1)
        # x.shape === (B, 2048+64, 1)
        x = x.repeat(1, 1, N)
        # x.shape == (B, 2048+64, N)
        x = torch.cat([x, x_tmp], dim=1)
        # x.shape == (B, 2048+64+C, N)
        x = self.conv2(x)
        # x.shape == (B, 1024, N)
        x = self.dp1(x)
        # x.shape == (B, 1024, N)
        x = self.conv3(x)
        # x.shape == (B, 256, N)
        x = self.dp2(x)
        # x.shape == (B, 256, N)
        x = self.conv4(x)
        # x.shape == (B, 50, N)

        if self.stn_regularization_loss_factor > 0:
            return x, trans
        else:
            return x
