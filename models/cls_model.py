import torch
from torch import nn
from models import cls_block
import time


class ModelNetModel(nn.Module):
    def __init__(self, config, calculate_inference_time=False, fps=False):

        super(ModelNetModel, self).__init__()

        if config.feature_learning_block.enable:
            self.block = cls_block.FeatureLearningBlock(config.feature_learning_block, calculate_inference_time, fps)
            num_layers = len(config.feature_learning_block.attention.K)
        else:
            raise ValueError('Only neighbor2point block supported!')

        num_output = 40

        self.res_link_enable = config.feature_learning_block.res_link.enable
        self.aux_loss_enable = config.train.aux_loss.enable
        self.aux_loss_shared = config.train.aux_loss.shared
        self.aux_loss_concat = config.train.aux_loss.concat
        consistency_loss_factor = config.train.consistency_loss_factor

        if self.res_link_enable:
            if self.aux_loss_enable:
                if self.aux_loss_shared:
                    self.linear1 = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512),
                                                 nn.LeakyReLU(negative_slope=0.2), nn.Dropout(p=0.5))
                    self.linear2 = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256),
                                                 nn.LeakyReLU(negative_slope=0.2), nn.Dropout(p=0.5))
                    self.linear3 = nn.Linear(256, num_output)
                else:
                    self.linear1_list = nn.ModuleList()
                    self.linear2_list = nn.ModuleList()
                    self.linear3_list = nn.ModuleList()
                    for i in range(num_layers):
                        self.linear1_list.append(
                            nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.LeakyReLU(negative_slope=0.2),
                                          nn.Dropout(p=0.5)))
                        self.linear2_list.append(
                            nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256), nn.LeakyReLU(negative_slope=0.2),
                                          nn.Dropout(p=0.5)))
                        self.linear3_list.append(nn.Linear(256, num_output))
                if self.aux_loss_concat:
                    self.linear0 = nn.Sequential(nn.Linear(1024 * num_layers, 1024), nn.BatchNorm1d(1024),
                                                 nn.LeakyReLU(negative_slope=0.2), nn.Dropout(p=0.5))
            else:
                self.linear1 = nn.Sequential(nn.Linear(1024 * num_layers, 1024), nn.BatchNorm1d(1024),
                                             nn.LeakyReLU(negative_slope=0.2), nn.Dropout(p=0.5))
                self.linear2 = nn.Sequential(nn.Linear(1024, 256), nn.BatchNorm1d(256),
                                             nn.LeakyReLU(negative_slope=0.2), nn.Dropout(p=0.5))
                self.linear3 = nn.Linear(256, num_output)
        else:
            assert self.aux_loss_enable == False and consistency_loss_factor == 0, "If there is no residual link in the structure, consistency loss and auxiliary loss must be False!"
            self.linear2 = nn.Sequential(nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.LeakyReLU(negative_slope=0.2),
                                         nn.Dropout(p=0.5))
            self.linear3 = nn.Linear(256, num_output)


    def forward(self, x):  # x.shape == (B, 3, N)
        if self.calculate_inference_time:
            torch.cuda.synchronize()
            self.beginning = time.time()

        if self.res_link_enable:
            # with res_link
            x, x_res_link_list = self.block(x)  # x.shape == (B, 3C)

            if self.calculate_inference_time:
                self.before_ds = self.block.before_ds
                self.after_fps = self.block.after_fps
                self.after_ds = self.block.after_ds

            if self.aux_loss_enable:
                # with aux loss
                x_aux_list = []
                if self.aux_loss_shared:
                    if self.aux_loss_concat:
                        # aux_shared-concat
                        for i, x_res in enumerate(x_res_link_list):
                            if i != len(x_res_link_list) - 1:
                                x_aux_list.append(self.MLP(x_res))
                        x_aux_list.append(self.MLP_concat(x))
                    else:
                        # aux_shared-unconcat
                        for x_res in x_res_link_list:
                            x_aux_list.append(self.MLP(x_res))
                else:
                    if self.aux_loss_concat:
                        # aux_unshared-concat
                        for i, x_res in enumerate(x_res_link_list):
                            if i != len(x_res_link_list) - 1:
                                x_aux_list.append(self.MLP_unshared(x_res, i))
                            else:
                                x_aux_list.append(self.MLP_unshared_concat(x, i))
                    else:
                        # aux_unshared-unconcat
                        for i, x_res in enumerate(x_res_link_list):
                            x_aux_list.append(self.MLP_unshared(x_res, i))
                return x_aux_list
            else:
                # no_aux
                x = self.MLP(x)  # x.shape == (B, 40)

                if self.calculate_inference_time:
                    torch.cuda.synchronize()
                    self.end_time = time.time()

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
