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


# ================================================================================
# Yi650M shapenet dataloader


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


def fps(x,xyz, npoint):
    xyz=torch.permute(xyz,(0,2,1))
    x=torch.permute(x,(0,2,1))

    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint]
    # new_xyz = index_points(xyz, fps_idx)
    x = index_points(x, fps_idx)

    # x(B,N,C)
    x=torch.permute(x,(0,2,1))
    # fps_idx(B,S)
    fps_idx=torch.unsqueeze(fps_idx,dim=1)

    return (x, fps_idx), (None,None)


def download_shapenet_Yi650M(url, saved_path):
    ssl._create_default_https_context = ssl._create_unverified_context
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    # check if dataset already exists
    path = Path(saved_path, "shapenetcore_partanno_segmentation_benchmark_v0")
    if not path.exists():
        print("Downloading dataset, please wait...")
        wget.download(url=url, out=saved_path)
        print()
        file = str(Path(saved_path, url.split("/")[-1]).resolve())
        print("Unpacking dataset, please wait...")
        shutil.unpack_archive(file, saved_path)
        os.remove(file)


class ShapeNet_Yi650M(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        json_path,
        mapping,
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
        self.root = root
        self.mapping = mapping
        self.selected_points = selected_points
        self.fps_enable = fps_enable
        self.augmentation = augmentation
        self.num_aug = num_aug
        self.vote = vote_enable
        self.vote_num = vote_num
        if len(json_path) == 1:
            file_name, _ = os.path.splitext(os.path.basename(json_path[0]))
            self.partition = file_name.split("_")[-3]
        else:
            self.partition = "trainval"
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
        self.samples = []
        for each_path in json_path:
            with open(each_path, "r") as f:
                self.samples.extend(json.load(f))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        category_hash, pcd_hash = sample.split("/")[1:]

        # get point cloud
        pcd_path = os.path.join(self.root, category_hash, "points", f"{pcd_hash}.pts")
        pcd = np.loadtxt(pcd_path)
        # get a fixed number of points from every point cloud
        if self.fps_enable:
            if self.selected_points <= len(pcd):
                pcd = torch.Tensor(
                    pcd[None, ...]
                ).cuda()  # fps requires batch size dimension
                # pcd, indices = fps(pcd, K=self.selected_points, random_start_point=True)
                # pcd B N C
                pcd, indices = fps(pcd, K=self.selected_points)
                pcd, indices = (
                    pcd[0].cpu().numpy(),
                    indices[0].cpu().numpy(),
                )  # squeeze the batch size dimension
            else:
                indices = np.random.choice(len(pcd), self.selected_points, replace=True)
                pcd = pcd[indices]
        else:
            if self.selected_points <= len(pcd):
                indices = np.random.choice(
                    len(pcd), self.selected_points, replace=False
                )
                pcd = pcd[indices]
            else:
                indices = np.random.choice(len(pcd), self.selected_points, replace=True)
                pcd = pcd[indices]

        # get point cloud augmentation or voting list
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
        parts_id = self.mapping[category_hash]["parts_id"]
        seg_label_path = os.path.join(
            self.root, category_hash, "points_label", f"{pcd_hash}.seg"
        )
        seg_label = np.loadtxt(seg_label_path).astype("float32")
        seg_label = seg_label[indices]
        # match parts id and convert seg label to one hot
        diff = min(parts_id) - 1
        seg_label = seg_label + diff
        seg_label = (
            F.one_hot(torch.Tensor(seg_label).long(), 50)
            .to(torch.float32)
            .permute(1, 0)
        )

        # get category one hot
        category_id = self.mapping[category_hash]["category_id"]
        category_onehot = (
            F.one_hot(torch.Tensor([category_id]).long(), 16)
            .to(torch.float32)
            .permute(1, 0)
        )

        # pcd.shape == (3, N)    seg_label.shape == (50, N)    category_onehot.shape == (16, 1)
        return pcd, seg_label, category_onehot


def get_shapenet_dataset_Yi650M(
    saved_path,
    mapping,
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
    dataset_path = Path(saved_path, "shapenetcore_partanno_segmentation_benchmark_v0")
    # get datasets json files
    train_json = os.path.join(
        dataset_path, "train_test_split", "shuffled_train_file_list.json"
    )
    validation_json = os.path.join(
        dataset_path, "train_test_split", "shuffled_val_file_list.json"
    )
    test_json = os.path.join(
        dataset_path, "train_test_split", "shuffled_test_file_list.json"
    )

    # get datasets
    train_set = ShapeNet_Yi650M(
        dataset_path,
        [train_json],
        mapping,
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
    validation_set = ShapeNet_Yi650M(
        dataset_path,
        [validation_json],
        mapping,
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
    trainval_set = ShapeNet_Yi650M(
        dataset_path,
        [train_json, validation_json],
        mapping,
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
    test_set = ShapeNet_Yi650M(
        dataset_path,
        [test_json],
        mapping,
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
# AnTao350M shapenet dataloader


def download_shapenet_AnTao350M(url, saved_path):
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


class ShapeNet_AnTao350M(torch.utils.data.Dataset):
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


def get_shapenet_dataset_AnTao350M(
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
    train_set = ShapeNet_AnTao350M(
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
    validation_set = ShapeNet_AnTao350M(
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
    trainval_set = ShapeNet_AnTao350M(
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
    test_set = ShapeNet_AnTao350M(
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


def download_modelnet_AnTao420M(url, saved_path):
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


class ModelNet_AnTao420M(torch.utils.data.Dataset):
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


def get_modelnet_dataset_AnTao420M(
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
    trainval_set = ModelNet_AnTao420M(
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
    test_set = ModelNet_AnTao420M(
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


# ================================================================================
# Alignment1024 modelnet dataloader


def download_modelnet_Alignment1024(url, saved_path):
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    dataset_path = os.path.join(saved_path, "modelnet40_normal_resampled")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        os.system(
            'gdown "https://drive.google.com/uc?id=1fq4G5djBblr6FME7TY5WH7Lnz9psVf4i"'
        )
        os.system(
            'gdown "https://drive.google.com/uc?id=1WzcIm2G55yTh-snOrdeiZJrYDBqJeAck"'
        )
        os.system("mv %s %s" % ("modelnet40_train_1024pts_fps.dat", dataset_path))
        os.system("mv %s %s" % ("modelnet40_test_1024pts_fps.dat", dataset_path))


class ModelNet_Alignment1024(torch.utils.data.Dataset):
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
    ):
        super(ModelNet_Alignment1024, self).__init__()
        self.selected_points = selected_points
        self.fps_enable = fps_enable
        self.augmentation = augmentation
        self.num_aug = num_aug
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
        if partition == "trainval":
            data_path = os.path.join(
                saved_path,
                "modelnet40_normal_resampled",
                "modelnet40_train_1024pts_fps.dat",
            )
        elif partition == "test":
            data_path = os.path.join(
                saved_path,
                "modelnet40_normal_resampled",
                "modelnet40_test_1024pts_fps.dat",
            )
        else:
            raise ValueError(
                "modelnet40 has only train_set and test_set, which means validation_set is included in train_set!"
            )
        with open(data_path, "rb") as f:
            self.all_pcd, self.all_cls_label = pickle.load(f)
        self.all_pcd = np.stack(self.all_pcd, axis=0)[:, :, :3]
        self.all_cls_label = np.stack(self.all_cls_label, axis=0)[:, 0]

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
            # indices = np.random.choice(1024, self.selected_points, False)
            indices = np.random.choice(len(pcd), self.selected_points, False)
            pcd = pcd[indices]
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


def get_modelnet_dataset_Alignment1024(
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
):
    # get dataset
    trainval_set = ModelNet_Alignment1024(
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
    test_set = ModelNet_Alignment1024(
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
    )
    return trainval_set, test_set


# ================================================================================
# Normal shapenet dataloader
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def download_shapenet_Normal(url, saved_path):
    ssl._create_default_https_context = ssl._create_unverified_context
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    # check if dataset already exists
    path = Path(saved_path, "shapenetcore_partanno_segmentation_benchmark_v0_normal")
    if not path.exists():
        print("Downloading dataset, please wait...")
        wget.download(url=url, out=saved_path)
        print()
        file = str(Path(saved_path, url.split("/")[-1]).resolve())
        print("Unpacking dataset, please wait...")
        shutil.unpack_archive(file, saved_path)
        os.remove(file)


class ShapeNet_Normal(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        json_path,
        mapping,
        config_dataloader,
        vote_enable=False,
        vote_num=10,
        normal_channel=True,
    ):
        self.root = root
        self.selected_points = config_dataloader.selected_points
        self.fps_enable = config_dataloader.fps
        self.vote = vote_enable
        self.vote_num = vote_num
        self.normal_channel = normal_channel
        self.mapping = mapping

        config_aug = config_dataloader.data_augmentation
        self.augmentation = config_aug.enable
        self.num_aug = config_aug.num_aug
        if self.augmentation:
            self.augmentation_list = []
            if config_aug.rotate_perturbation.enable:
                self.augmentation_list.append(
                    [
                        data_augmentation.rotate_perturbation_with_normal,
                        [
                            config_aug.rotate_perturbation.std,
                            config_aug.rotate_perturbation.clip,
                        ],
                    ]
                )
            if config_aug.rotate.enable:
                self.augmentation_list.append(
                    [
                        data_augmentation.rotate_with_normal,
                        [config_aug.rotate.angle_range],
                    ]
                )
            if config_aug.translate.enable:
                self.augmentation_list.append(
                    [
                        data_augmentation.translate,
                        [
                            config_aug.translate.x_range,
                            config_aug.translate.y_range,
                            config_aug.translate.z_range,
                            self.normal_channel,
                        ],
                    ]
                )
            if config_aug.anisotropic_scale.enable:
                self.augmentation_list.append(
                    [
                        data_augmentation.anisotropic_scale,
                        [
                            config_aug.anisotropic_scale.x_range,
                            config_aug.anisotropic_scale.y_range,
                            config_aug.anisotropic_scale.z_range,
                            True,
                            self.normal_channel,
                        ],
                    ]
                )
            if (
                not config_aug.rotate_perturbation.enable
                and not config_aug.rotate.enable
                and not config_aug.translate.enable
                and not config_aug.anisotropic_scale.enable
            ):
                raise ValueError(
                    "At least one kind of data augmentation should be applied!"
                )
            if len(self.augmentation_list) < self.num_aug:
                raise ValueError(
                    f"num_aug should not be less than the number of enabled augmentations. num_aug: {config_aug.num_aug}, number of enabled augmentations: {len(self.augmentation_list)}"
                )

        if len(json_path) == 1:
            file_name, _ = os.path.splitext(os.path.basename(json_path[0]))
            self.partition = file_name.split("_")[-3]
        else:
            self.partition = "trainval"
        if self.vote:
            self.vote_list = []
            for _ in range(self.vote_num - 1):
                self.vote_list.append(
                    [
                        data_augmentation.anisotropic_scale,
                        [
                            config_aug.anisotropic_scale.x_range,
                            config_aug.anisotropic_scale.y_range,
                            config_aug.anisotropic_scale.z_range,
                            True,
                            self.normal_channel,
                        ],
                    ]
                )
        self.samples = []
        for each_path in json_path:
            with open(each_path, "r") as f:
                self.samples.extend(json.load(f))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        category_hash, pcd_hash = sample.split("/")[1:]

        # get point cloud
        data_path = os.path.join(self.root, category_hash, f"{pcd_hash}.txt")
        data = np.loadtxt(data_path).astype("float32")
        if self.normal_channel:
            pcd = data[:, :6]
        else:
            pcd = data[:, :3]

        # get a fixed number of points from every point cloud
        if self.fps_enable:
            if self.selected_points <= len(pcd):
                pcd_fps = pcd[:, :3]
                pcd_fps = torch.Tensor(
                    pcd_fps[None, ...]
                ).cuda()  # fps requires batch size dimension
                _, indices = fps(
                    pcd_fps, K=self.selected_points
                )  # , random_start_point=True)
                # get pcd from indices
                indices = indices[0].cpu().numpy()  # squeeze the batch size dimension
                pcd = pcd[indices]
            else:
                indices = np.random.choice(len(pcd), self.selected_points, replace=True)
                pcd = pcd[indices]
        else:
            if self.selected_points <= len(pcd):
                indices = np.random.choice(
                    len(pcd), self.selected_points, replace=False
                )
                pcd = pcd[indices]
            else:
                indices = np.random.choice(len(pcd), self.selected_points, replace=True)
                pcd = pcd[indices]

        # get point cloud augmentation or voting list
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
        seg_label = data[:, -1]
        seg_label = seg_label[indices]
        # match parts id and convert seg label to one hot
        seg_label = (
            F.one_hot(torch.Tensor(seg_label).long(), 50)
            .to(torch.float32)
            .permute(1, 0)
        )

        # get category one hot
        category_id = self.mapping[category_hash]["category_id"]
        category_onehot = (
            F.one_hot(torch.Tensor([category_id]).long(), 16)
            .to(torch.float32)
            .permute(1, 0)
        )

        # pcd.shape == (3, N)    seg_label.shape == (50, N)    category_onehot.shape == (16, 1)
        return pcd, seg_label, category_onehot


def get_shapenet_dataset_Normal(config_datasets, config_dataloader, config_vote):
    saved_path = config_datasets.saved_path
    mapping = config_datasets.mapping
    vote_enable = config_vote.enable
    vote_num = config_vote.num_vote

    dataset_path = Path(
        saved_path, "shapenetcore_partanno_segmentation_benchmark_v0_normal"
    )
    # get datasets json files
    train_json = os.path.join(
        dataset_path, "train_test_split", "shuffled_train_file_list.json"
    )
    validation_json = os.path.join(
        dataset_path, "train_test_split", "shuffled_val_file_list.json"
    )
    test_json = os.path.join(
        dataset_path, "train_test_split", "shuffled_test_file_list.json"
    )

    # get datasets
    train_set = ShapeNet_Normal(dataset_path, [train_json], mapping, config_dataloader)
    validation_set = ShapeNet_Normal(
        dataset_path, [validation_json], mapping, config_dataloader
    )
    trainval_set = ShapeNet_Normal(
        dataset_path, [train_json, validation_json], mapping, config_dataloader
    )
    test_set = ShapeNet_Normal(
        dataset_path, [test_json], mapping, config_dataloader, vote_enable, vote_num
    )
    return train_set, validation_set, trainval_set, test_set
