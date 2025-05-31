# *_*coding:utf-8 *_*
import ssl
import shutil
import wget
from pathlib import Path
import os
import torch
import torch.nn.functional as F
import json
import numpy as np
import h5py
import glob
from utils import data_augmentation

# from pytorch3d.ops import sample_farthest_points as fps
# from openpoints.models.layers.subsample import fps
import pickle
import warnings
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B, N, S = points.shape
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def fps(x, xyz, npoint):
    xyz = torch.permute(xyz, (0, 2, 1))
    x = torch.permute(x, (0, 2, 1))

    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint]
    # new_xyz = index_points(xyz, fps_idx)
    x = index_points(x, fps_idx)

    # x(B,N,C)
    x = torch.permute(x, (0, 2, 1))
    # fps_idx(B,S)
    fps_idx = torch.unsqueeze(fps_idx, dim=1)

    return (x, fps_idx), (None, None)


# ================================================================================
# AnTao350M shapenet dataloader


def download_shapenet(url, saved_path):
    # current_directory = os.getcwd()
    # print(current_directory)
    # saved_path_0 = os.path.join(saved_path, 'shapenet_part_seg_hdf5_data')
    # print(saved_path_0)
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    if not os.path.exists(os.path.join(saved_path, "shapenet_part_seg_hdf5_data")):
        zipfile = os.path.basename(url)
        os.system("wget %s --no-check-certificate; unzip %s" % (url, zipfile))

        os.system(
            "mv %s %s"
            % ("hdf5_data", os.path.join(saved_path, "shapenet_part_seg_hdf5_data"))
        )
        # command0='wget %s --no-check-certificate; unzip %s' % (url, zipfile)
        # command = 'mv %s %s' % ('hdf5_data', os.path.join(saved_path, 'shapenet_part_seg_hdf5_data'))
        os.system("rm %s" % (zipfile))


class ShapeNet(torch.utils.data.Dataset):
    def __init__(
        self,
        saved_path,
        partition,
        selected_points,
        fps_enable,
        augmentation,
        num_aug,
        jitter,
        std,
        clip,
        rotate,
        which_axis,
        angle_range,
        translate,
        x_translate_range,
        y_translate_range,
        z_translate_range,
        anisotropic_scale,
        x_scale_range,
        y_scale_range,
        z_scale_range,
        isotropic,
        vote_enable=False,
        vote_num=10,
    ):
        self.selected_points = selected_points
        self.fps_enable = fps_enable
        self.augmentation = augmentation
        self.num_aug = num_aug
        self.partition = partition
        self.vote = vote_enable
        self.vote_num = vote_num
        if augmentation:
            self.augmentation_list = []
            if jitter:
                self.augmentation_list.append([data_augmentation.jitter, [std, clip]])
            if rotate:
                self.augmentation_list.append(
                    [data_augmentation.rotate, [which_axis, angle_range]]
                )
            if translate:
                self.augmentation_list.append(
                    [
                        data_augmentation.translate,
                        [x_translate_range, y_translate_range, z_translate_range],
                    ]
                )
            if anisotropic_scale:
                self.augmentation_list.append(
                    [
                        data_augmentation.anisotropic_scale,
                        [x_scale_range, y_scale_range, z_scale_range, isotropic],
                    ]
                )
            if not jitter and not rotate and not translate and not anisotropic_scale:
                raise ValueError(
                    "At least one kind of data augmentation should be applied!"
                )
            if len(self.augmentation_list) < num_aug:
                raise ValueError(
                    f"num_aug should not be less than the number of enabled augmentations. num_aug: {num_aug}, number of enabled augmentations: {len(self.augmentation_list)}"
                )
        if self.vote:
            self.vote_list = []
            for _ in range(self.vote_num - 1):
                self.vote_list.append(
                    [
                        data_augmentation.anisotropic_scale,
                        [x_scale_range, y_scale_range, z_scale_range, isotropic],
                    ]
                )
        self.all_pcd = []
        self.all_cls_label = []
        self.all_seg_label = []
        if partition == "trainval":
            file = glob.glob(
                os.path.join(saved_path, "shapenet_part_seg_hdf5_data", "*train*.h5")
            ) + glob.glob(
                os.path.join(saved_path, "shapenet_part_seg_hdf5_data", "*val*.h5")
            )
            file.sort()
        else:
            file = glob.glob(
                os.path.join(
                    saved_path, "shapenet_part_seg_hdf5_data", "*%s*.h5" % partition
                )
            )
            file.sort()
        for h5_name in file:
            f = h5py.File(h5_name, "r+")
            pcd = f["data"][:].astype("float32")
            cls_label = f["label"][:].astype("int64")
            seg_label = f["pid"][:].astype("int64")
            f.close()
            self.all_pcd.append(pcd)
            self.all_cls_label.append(cls_label)
            self.all_seg_label.append(seg_label)
        self.all_pcd = np.concatenate(self.all_pcd, axis=0)
        self.all_cls_label = np.concatenate(self.all_cls_label, axis=0)
        self.all_seg_label = np.concatenate(self.all_seg_label, axis=0)

    def __len__(self):
        return self.all_cls_label.shape[0]

    def __getitem__(self, index):
        # get category one hot
        category_id = self.all_cls_label[index, 0]
        category_onehot = (
            F.one_hot(torch.Tensor([category_id]).long(), 16)
            .to(torch.float32)
            .permute(1, 0)
        )

        # get point cloud
        pcd = self.all_pcd[index]
        if self.fps_enable:
            pcd = torch.Tensor(
                pcd[None, ...]
            ).cuda()  # fps requires batch size dimension
            pcd, indices = fps(
                pcd, K=self.selected_points
            )  # , random_start_point=True)
            pcd, indices = (
                pcd[0].cpu().numpy(),
                indices[0].cpu().numpy(),
            )  # squeeze the batch size dimension
        else:
            # shuffle points within one point cloud
            indices = np.random.choice(2048, self.selected_points, False)
            pcd = pcd[indices]

        if self.partition == "test" and self.vote:
            pcd_tmp_list = []
            pcd_list = []
            for i in range(len(self.vote_list)):
                augmentation, params = self.vote_list[i]
                pcd_tmp = augmentation(pcd, *params)
                pcd_tmp_list.append(pcd_tmp)
            for i, pcd_tmp in enumerate(pcd_tmp_list):
                if i == 0:
                    pcd = torch.Tensor(pcd).to(torch.float32)
                else:
                    pcd = torch.Tensor(pcd_tmp).to(torch.float32)
                pcd = pcd.permute(1, 0)
                pcd_list.append(pcd)
            pcd = pcd_list
        else:
            if self.augmentation:
                choice = np.random.choice(
                    len(self.augmentation_list), self.num_aug, replace=False
                )
                for i in choice:
                    augmentation, params = self.augmentation_list[i]
                    pcd = augmentation(pcd, *params)
            pcd = torch.Tensor(pcd).to(torch.float32)
            pcd = pcd.permute(1, 0)

        # get point cloud seg label
        seg_label = self.all_seg_label[index].astype("float32")
        seg_label = seg_label[indices]
        # match parts id and convert seg label to one hot
        seg_label = (
            F.one_hot(torch.Tensor(seg_label).long(), 50)
            .to(torch.float32)
            .permute(1, 0)
        )

        # pcd.shape == (3, N)    seg_label.shape == (50, N)    category_onehot.shape == (16, 1)
        return pcd, seg_label, category_onehot


def get_shapenet_dataset(
    saved_path,
    selected_points,
    fps_enable,
    augmentation,
    num_aug,
    jitter,
    std,
    clip,
    rotate,
    which_axis,
    angle_range,
    translate,
    x_translate_range,
    y_translate_range,
    z_translate_range,
    anisotropic_scale,
    x_scale_range,
    y_scale_range,
    z_scale_range,
    isotropic,
    vote_enable=False,
    vote_num=10,
):
    # get dataset
    train_set = ShapeNet(
        saved_path,
        "train",
        selected_points,
        fps_enable,
        augmentation,
        num_aug,
        jitter,
        std,
        clip,
        rotate,
        which_axis,
        angle_range,
        translate,
        x_translate_range,
        y_translate_range,
        z_translate_range,
        anisotropic_scale,
        x_scale_range,
        y_scale_range,
        z_scale_range,
        isotropic,
    )
    validation_set = ShapeNet(
        saved_path,
        "val",
        selected_points,
        fps_enable,
        False,
        num_aug,
        jitter,
        std,
        clip,
        rotate,
        which_axis,
        angle_range,
        translate,
        x_translate_range,
        y_translate_range,
        z_translate_range,
        anisotropic_scale,
        x_scale_range,
        y_scale_range,
        z_scale_range,
        isotropic,
    )
    trainval_set = ShapeNet(
        saved_path,
        "trainval",
        selected_points,
        fps_enable,
        augmentation,
        num_aug,
        jitter,
        std,
        clip,
        rotate,
        which_axis,
        angle_range,
        translate,
        x_translate_range,
        y_translate_range,
        z_translate_range,
        anisotropic_scale,
        x_scale_range,
        y_scale_range,
        z_scale_range,
        isotropic,
    )
    test_set = ShapeNet(
        saved_path,
        "test",
        selected_points,
        fps_enable,
        False,
        num_aug,
        jitter,
        std,
        clip,
        rotate,
        which_axis,
        angle_range,
        translate,
        x_translate_range,
        y_translate_range,
        z_translate_range,
        anisotropic_scale,
        x_scale_range,
        y_scale_range,
        z_scale_range,
        isotropic,
        vote_enable,
        vote_num,
    )
    return train_set, validation_set, trainval_set, test_set


# ================================================================================
# AnTao420M modelnet dataloader


def download_modelnet(url, saved_path):
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    if not os.path.exists(os.path.join(saved_path, "modelnet40_ply_hdf5_2048")):
        zipfile = os.path.basename(url)
        os.system("wget %s --no-check-certificate; unzip %s" % (url, zipfile))
        os.system(
            "mv %s %s"
            % (
                "modelnet40_ply_hdf5_2048",
                os.path.join(saved_path, "modelnet40_ply_hdf5_2048"),
            )
        )
        os.system("rm %s" % (zipfile))


class ModelNet(torch.utils.data.Dataset):
    def __init__(
        self,
        saved_path,
        partition,
        selected_points,
        fps_enable,
        augmentation,
        num_aug,
        jitter,
        std,
        clip,
        rotate,
        which_axis,
        angle_range,
        translate,
        x_translate_range,
        y_translate_range,
        z_translate_range,
        anisotropic_scale,
        x_scale_range,
        y_scale_range,
        z_scale_range,
        isotropic,
        vote_enable=False,
        vote_num=10,
    ):
        self.selected_points = selected_points
        self.fps_enable = fps_enable
        self.augmentation = augmentation
        self.num_aug = num_aug
        self.vote = vote_enable
        self.vote_num = vote_num
        self.partition = partition

        if augmentation:
            self.augmentation_list = []
            if jitter:
                self.augmentation_list.append([data_augmentation.jitter, [std, clip]])
            if rotate:
                self.augmentation_list.append(
                    [data_augmentation.rotate, [which_axis, angle_range]]
                )
            if translate:
                self.augmentation_list.append(
                    [
                        data_augmentation.translate,
                        [x_translate_range, y_translate_range, z_translate_range],
                    ]
                )
            if anisotropic_scale:
                self.augmentation_list.append(
                    [
                        data_augmentation.anisotropic_scale,
                        [x_scale_range, y_scale_range, z_scale_range, isotropic],
                    ]
                )
            if not jitter and not rotate and not translate and not anisotropic_scale:
                raise ValueError(
                    "At least one kind of data augmentation should be applied!"
                )
            if len(self.augmentation_list) < num_aug:
                raise ValueError(
                    f"num_aug should not be less than the number of enabled augmentations. num_aug: {num_aug}, number of enabled augmentations: {len(self.augmentation_list)}"
                )
        self.all_pcd = []
        self.all_cls_label = []
        if partition == "trainval":
            file = glob.glob(
                os.path.join(saved_path, "modelnet40_ply_hdf5_2048", "*train*.h5")
            )
            file.sort()
        elif partition == "test":
            file = glob.glob(
                os.path.join(saved_path, "modelnet40_ply_hdf5_2048", "*test*.h5")
            )
            file.sort()

            if self.vote:
                self.vote_list = []
                for _ in range(self.vote_num - 1):
                    self.vote_list.append(
                        [
                            data_augmentation.anisotropic_scale,
                            [x_scale_range, y_scale_range, z_scale_range, isotropic],
                        ]
                    )
        else:
            raise ValueError(
                "modelnet40 has only train_set and test_set, which means validation_set is included in train_set!"
            )
        for h5_name in file:
            f = h5py.File(h5_name, "r+")
            pcd = f["data"][:].astype("float32")
            cls_label = f["label"][:].astype("int64")
            f.close()
            self.all_pcd.append(pcd)
            self.all_cls_label.append(cls_label[:, 0])
        self.all_pcd = np.concatenate(self.all_pcd, axis=0)
        self.all_cls_label = np.concatenate(self.all_cls_label, axis=0)

    def __len__(self):
        return self.all_cls_label.shape[0]

    def __getitem__(self, index):
        # get category one hot
        category_id = self.all_cls_label[index]
        category_onehot = (
            F.one_hot(torch.Tensor([category_id]).long(), 40)
            .to(torch.float32)
            .squeeze()
        )

        # get point cloud
        pcd = self.all_pcd[index]
        if self.fps_enable:
            pcd = torch.Tensor(
                pcd[None, ...]
            ).cuda()  # fps requires batch size dimension
            pcd, _ = fps(pcd, K=self.selected_points)  # , random_start_point=True)
            pcd = pcd[0].cpu().numpy()  # squeeze the batch size dimension
        else:
            indices = np.random.choice(2048, self.selected_points, False)
            pcd = pcd[indices]

        if self.partition == "test" and self.vote:
            pcd_tmp_list = []
            pcd_list = []
            for i in range(len(self.vote_list)):
                augmentation, params = self.vote_list[i]
                pcd_tmp = augmentation(pcd, *params)
                pcd_tmp_list.append(pcd_tmp)
            for i, pcd_tmp in enumerate(pcd_tmp_list):
                if i == 0:
                    pcd = torch.Tensor(pcd).to(torch.float32)
                else:
                    pcd = torch.Tensor(pcd_tmp).to(torch.float32)
                pcd = pcd.permute(1, 0)
                pcd_list.append(pcd)
            pcd = pcd_list
        else:
            if self.augmentation:
                choice = np.random.choice(
                    len(self.augmentation_list), self.num_aug, replace=False
                )
                for i in choice:
                    augmentation, params = self.augmentation_list[i]
                    pcd = augmentation(pcd, *params)

            pcd = torch.Tensor(pcd).to(torch.float32)
            pcd = pcd.permute(1, 0)

        # pcd.shape == (C, N)  category_onehot.shape == (40,)
        return pcd, category_onehot


def get_modelnet_dataset(
    saved_path,
    selected_points,
    fps_enable,
    augmentation,
    num_aug,
    jitter,
    std,
    clip,
    rotate,
    which_axis,
    angle_range,
    translate,
    x_translate_range,
    y_translate_range,
    z_translate_range,
    anisotropic_scale,
    x_scale_range,
    y_scale_range,
    z_scale_range,
    isotropic,
    vote_enable=False,
    vote_num=10,
):
    # get dataset
    trainval_set = ModelNet(
        saved_path,
        "trainval",
        selected_points,
        fps_enable,
        augmentation,
        num_aug,
        jitter,
        std,
        clip,
        rotate,
        which_axis,
        angle_range,
        translate,
        x_translate_range,
        y_translate_range,
        z_translate_range,
        anisotropic_scale,
        x_scale_range,
        y_scale_range,
        z_scale_range,
        isotropic,
    )
    test_set = ModelNet(
        saved_path,
        "test",
        selected_points,
        fps_enable,
        False,
        num_aug,
        jitter,
        std,
        clip,
        rotate,
        which_axis,
        angle_range,
        translate,
        x_translate_range,
        y_translate_range,
        z_translate_range,
        anisotropic_scale,
        x_scale_range,
        y_scale_range,
        z_scale_range,
        isotropic,
        vote_enable,
        vote_num,
    )
    return trainval_set, test_set
