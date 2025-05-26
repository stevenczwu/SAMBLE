import shutil
import numpy as np
import pkbar
import math
from plyfile import PlyData, PlyElement
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
from collections import OrderedDict
import pickle
from tqdm import tqdm
import copy
from PIL import Image

from .ops import calculate_num_points_to_choose
from .visualization_data_processing import *
import torch


def visualization_heatmap_one_shape(i, sample, category, atten, save_path, view_range):
    # make every category name start from 0
    my_cmap = cm.get_cmap('viridis_r', sample.shape[0])
    # print(f'sample.shape{sample.shape}')
    xyzRGB = []

    # atten = np.log(atten)
    atten = (atten - np.mean(atten)) / np.std(atten) + 0.5

    # x_norm = (x - torch.min(x, dim=dim, keepdim=True)[0]) / (
    #         torch.max(x, dim=dim, keepdim=True)[0] - torch.min(x, dim=dim, keepdim=True)[0] + 1e-8)

    # atten = atten[0]
    # if mode == "trained_boltzmann":
    # atten = np.log(atten)
    # atten = atten - np.mean(atten) + 0.5
    # else:
    #     atten = (atten - np.mean(atten)) / np.std(atten) + 0.5

    for xyz, rgb in zip(sample, atten):
        # print(f'rgb:{rgb.shape}')
        xyzRGB_tmp = []
        xyzRGB_tmp.extend(list(xyz))
        # print(my_cmap)
        # print(np.asarray(my_cmap(rgb)))
        RGB = 255 * np.asarray(my_cmap(rgb))[:3]
        # print(f'RGB:{RGB}')
        # print(f'xyz.shape:{xyz.shape}')
        xyzRGB_tmp.extend(list(RGB))
        # print(f'xyzRGB_tmp:{xyzRGB_tmp}')
        xyzRGB.append(xyzRGB_tmp)

    if not os.path.exists(f'{save_path}/{category}/'):
        os.makedirs(f'{save_path}/{category}/')
    save_path = f'{save_path}/{category}/sample{i}_layer_0.png'

    vertex = np.array(xyzRGB)
    # print(f'vertex.shape:{vertex.shape}')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ax.set_xlim3d(-0.6, 0.6)
    # ax.set_ylim3d(-0.6, 0.6)
    # ax.set_zlim3d(-0.6, 0.6)
    ax.set_xlim3d(-view_range, view_range)
    ax.set_ylim3d(-view_range, view_range)
    ax.set_zlim3d(-view_range, view_range)
    ax.scatter(vertex[:, 0], vertex[:, 2], vertex[:, 1], c=vertex[:, 3:] / 255, marker='o', s=1)
    plt.axis('off')
    plt.grid('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    # print(f'.png file is saved in {saved_path}')


def visualize_shapenet_predictions(config, samples, preds, seg_labels, cls_label, shape_ious, index, artifacts_path):
    base_path = f'{artifacts_path}/vis_pred'
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    # get category name
    category_id_to_hash_code_mapping = {}
    for hash_code in list(config.datasets.mapping.keys()):
        category_id_to_hash_code_mapping[str(config.datasets.mapping[hash_code]['category_id'])] = hash_code
    categories = []
    for cat_id in cls_label:
        hash_code = category_id_to_hash_code_mapping[str(cat_id)]
        categories.append(config.datasets.mapping[hash_code]['category'])
    # select predictions
    samples_tmp = []
    preds_tmp = []
    seg_gts_tmp = []
    categories_tmp = []
    ious_tmp = []
    if config.edgeconv_block.enable:
        idx_tmp = [[] for _ in range(len(config.edgeconv_with_ds_block.samble_downsample.M))]
    if config.neighbor2point_block.enable:
        idx_tmp = [[] for _ in range(len(config.neighbor2point_block.samble_downsample.M))]
    if config.point2point_block.enable:
        idx_tmp = [[] for _ in range(len(config.point2point_block.samble_downsample.M))]
    i_categories = []
    for cat_id in config.test.visualize_preds.vis_which:
        samples_toappend = samples[cls_label == cat_id][:config.test.visualize_preds.num_vis]
        i_categories.append(samples_toappend.shape[0])
        samples_tmp.append(samples_toappend)
        preds_tmp.append(preds[cls_label == cat_id][:config.test.visualize_preds.num_vis])
        seg_gts_tmp.append(seg_labels[cls_label == cat_id][:config.test.visualize_preds.num_vis])
        categories_tmp.append(np.asarray(categories)[cls_label == cat_id][:config.test.visualize_preds.num_vis])
        ious_tmp.append(np.asarray(shape_ious)[cls_label == cat_id][:config.test.visualize_preds.num_vis])
        for layer, idx in enumerate(index):
            idx_tmp[layer].append(idx[cls_label == cat_id][:config.test.visualize_preds.num_vis])
    samples = np.concatenate(samples_tmp)
    preds = np.concatenate(preds_tmp)
    seg_labels = np.concatenate(seg_gts_tmp)
    categories = np.concatenate(categories_tmp)
    shape_ious = np.concatenate(ious_tmp)
    index = []
    for each in idx_tmp:
        index.append(np.concatenate(each))
    # start visualization
    pbar = pkbar.Pbar(name='Generating visualized prediction files, please wait...', target=len(samples))
    i_saved = 0
    if config.datasets.dataset_name == "shapenet_AnTao350M":
        view_range = 0.6
    elif config.datasets.dataset_name == "shapenet_Yi650M" or config.datasets.dataset_name == "shapenet_Normal":
        view_range = 0.3
    else:
        raise ValueError(f'Unknown dataset name: {config.datasets.dataset_name}')
    for i, (sample, pred, seg_gt, category, iou) in enumerate(zip(samples, preds, seg_labels, categories, shape_ious)):
        # make every category name start from 0
        for num_category in i_categories:
            if i_saved + num_category > i:
                break
            i_saved += num_category
        xyzRGB = []
        xyzRGB_gt = []
        xyzRGB_list = []
        xyzRGB_gt_list = []
        cat_path = f'{base_path}/{category}'
        if not os.path.exists(cat_path):
            os.makedirs(cat_path)
        for xyz, p, gt in zip(sample, pred, seg_gt):
            xyzRGB_tmp = []
            xyzRGB_gt_tmp = []
            xyzRGB_tmp.extend(list(xyz))
            xyzRGB_tmp.extend(config.datasets.cmap[str(p)])
            xyzRGB.append(tuple(xyzRGB_tmp))
            xyzRGB_gt_tmp.extend(list(xyz))
            xyzRGB_gt_tmp.extend(config.datasets.cmap[str(gt)])
            xyzRGB_gt.append(tuple(xyzRGB_gt_tmp))
        xyzRGB_list.append(xyzRGB)
        xyzRGB_gt_list.append(xyzRGB_gt)
        for layer, idx in enumerate(index):  # idx.shape == (B, H, K)
            xyzRGB = []
            xyzRGB_gt = []
            if layer != 0:
                idx = idx[i, 0]  # only visualize the first head
                for j in range(1, layer + 1):
                    idx = index[layer - j][i, 0][idx]  # index mapping
            else:
                idx = index[layer][i, 0]
            for xyz, p, gt in zip(sample[idx], pred[idx], seg_gt[idx]):
                xyzRGB_tmp = []
                xyzRGB_gt_tmp = []
                xyzRGB_tmp.extend(list(xyz))
                xyzRGB_tmp.extend(config.datasets.cmap[str(p)])
                xyzRGB.append(tuple(xyzRGB_tmp))
                xyzRGB_gt_tmp.extend(list(xyz))
                xyzRGB_gt_tmp.extend(config.datasets.cmap[str(gt)])
                xyzRGB_gt.append(tuple(xyzRGB_gt_tmp))
            xyzRGB_list.append(xyzRGB)
            xyzRGB_gt_list.append(xyzRGB_gt)
        if config.test.visualize_preds.format == 'ply':
            for which_layer, (xyzRGB, xyzRGB_gt) in enumerate(zip(xyzRGB_list, xyzRGB_gt_list)):
                if which_layer > 0:
                    pred_saved_path = f'{cat_path}/{category}{i - i_saved}_pred_{math.floor(iou * 1e5)}_dsLayer{which_layer}.ply'
                    gt_saved_path = f'{cat_path}/{category}{i - i_saved}_gt_dsLayer{which_layer}.ply'
                else:
                    pred_saved_path = f'{cat_path}/{category}{i - i_saved}_pred_{math.floor(iou * 1e5)}.ply'
                    gt_saved_path = f'{cat_path}/{category}{i - i_saved}_gt.ply'
                vertex = PlyElement.describe(np.array(xyzRGB,
                                                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'),
                                                             ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                PlyData([vertex]).write(pred_saved_path)
                vertex = PlyElement.describe(np.array(xyzRGB_gt,
                                                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'),
                                                             ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                PlyData([vertex]).write(gt_saved_path)
        elif config.test.visualize_preds.format == 'png':
            for which_layer, (xyzRGB, xyzRGB_gt) in enumerate(zip(xyzRGB_list, xyzRGB_gt_list)):
                if which_layer > 0:
                    pred_saved_path = f'{cat_path}/{category}{i - i_saved}_pred_{math.floor(iou * 1e5)}_dsLayer{which_layer}.png'
                    gt_saved_path = f'{cat_path}/{category}{i - i_saved}_gt_dsLayer{which_layer}.png'
                else:
                    pred_saved_path = f'{cat_path}/{category}{i - i_saved}_pred_{math.floor(iou * 1e5)}.png'
                    gt_saved_path = f'{cat_path}/{category}{i - i_saved}_gt.png'
                vertex = np.array(xyzRGB)
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.set_xlim3d(-view_range, view_range)
                ax.set_ylim3d(-view_range, view_range)
                ax.set_zlim3d(-view_range, view_range)
                ax.scatter(vertex[:, 0], vertex[:, 2], vertex[:, 1], c=vertex[:, 3:] / 255, marker='o', s=1)
                plt.axis('off')
                plt.grid('off')
                plt.savefig(pred_saved_path, bbox_inches='tight')
                plt.close(fig)
                vertex = np.array(xyzRGB_gt)
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.set_xlim3d(-view_range, view_range)
                ax.set_ylim3d(-view_range, view_range)
                ax.set_zlim3d(-view_range, view_range)
                ax.scatter(vertex[:, 0], vertex[:, 2], vertex[:, 1], c=vertex[:, 3:] / 255, marker='o', s=1)
                plt.axis('off')
                plt.grid('off')
                plt.savefig(gt_saved_path, bbox_inches='tight')
                plt.close(fig)
        else:
            raise ValueError(f'format should be png or ply, but got {config.test.visualize_preds.format}')
        pbar.update(i)
    print(f'Done! All files are saved in {base_path}')


def visualize_shapenet_downsampled_points(config, samples, index, cls_label, shape_ious, artifacts_path):
    base_path = f'{artifacts_path}/vis_ds_points'
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    # get category name
    category_id_to_hash_code_mapping = {}
    for hash_code in list(config.datasets.mapping.keys()):
        category_id_to_hash_code_mapping[str(config.datasets.mapping[hash_code]['category_id'])] = hash_code
    categories = []
    for cat_id in cls_label:
        hash_code = category_id_to_hash_code_mapping[str(cat_id)]
        categories.append(config.datasets.mapping[hash_code]['category'])
    # select samples
    samples_tmp = []
    categories_tmp = []
    ious_tmp = []
    if config.edgeconv_block.enable:
        idx_tmp = [[] for _ in range(len(config.edgeconv_with_ds_block.samble_downsample.M))]
    if config.neighbor2point_block.enable:
        idx_tmp = [[] for _ in range(len(config.neighbor2point_block.samble_downsample.M))]
    if config.point2point_block.enable:
        idx_tmp = [[] for _ in range(len(config.point2point_block.samble_downsample.M))]
    i_categories = []
    for cat_id in config.test.visualize_downsampled_points.vis_which:
        samples_toappend = samples[cls_label == cat_id][:config.test.visualize_downsampled_points.num_vis]
        i_categories.append(samples_toappend.shape[0])
        samples_tmp.append(samples_toappend)
        categories_tmp.append(
            np.asarray(categories)[cls_label == cat_id][:config.test.visualize_downsampled_points.num_vis])
        ious_tmp.append(np.asarray(shape_ious)[cls_label == cat_id][:config.test.visualize_downsampled_points.num_vis])
        for layer, idx in enumerate(index):
            idx_tmp[layer].append(idx[cls_label == cat_id][:config.test.visualize_downsampled_points.num_vis])
    samples = np.concatenate(samples_tmp)
    categories = np.concatenate(categories_tmp)
    shape_ious = np.concatenate(ious_tmp)
    index = []
    for each in idx_tmp:
        index.append(np.concatenate(each))
    # start visualization
    pbar = pkbar.Pbar(name='Generating visualized downsampled points files, please wait...', target=len(samples))
    i_saved = 0
    if config.datasets.dataset_name == "shapenet_AnTao350M":
        view_range = 0.6
    elif config.datasets.dataset_name == "shapenet_Yi650M" or config.datasets.dataset_name == "shapenet_Normal":
        view_range = 0.3
    else:
        raise ValueError(f'Unknown dataset name: {config.datasets.dataset_name}')
    for i, (sample, category, iou) in enumerate(zip(samples, categories, shape_ious)):
        # make every category name start from 0
        for num_category in i_categories:
            if i_saved + num_category > i:
                break
            i_saved += num_category
        xyzRGB = []
        for xyz in sample:
            xyzRGB_tmp = []
            xyzRGB_tmp.extend(list(xyz))
            xyzRGB_tmp.extend([192, 192, 192])  # gray color
            xyzRGB.append(xyzRGB_tmp)
        cat_path = f'{base_path}/{category}'
        if not os.path.exists(cat_path):
            os.makedirs(cat_path)
        for layer, idx in enumerate(index):  # idx.shape == (B, H, K)
            if layer != 0:
                idx = idx[i, 0]  # only visualize the first head
                for j in range(1, layer + 1):
                    idx = index[layer - j][i, 0][idx]  # index mapping
            else:
                idx = index[layer][i, 0]
            xyzRGB_tmp = deepcopy(xyzRGB)
            ds_tmp = []
            rst_tmp = []
            for each_idx in idx:
                xyzRGB_tmp[each_idx][3:] = [255, 0, 0]  # red color
                ds_tmp.append(xyzRGB_tmp[each_idx])
            for each in xyzRGB_tmp:
                if each not in ds_tmp:
                    rst_tmp.append(each)
            if config.test.visualize_downsampled_points.format == 'ply':
                saved_path = f'{cat_path}/{category}{i - i_saved}_layer{layer}_{math.floor(iou * 1e5)}.ply'
                vertex = PlyElement.describe(np.array(xyzRGB_tmp,
                                                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'),
                                                             ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                PlyData([vertex]).write(saved_path)
            elif config.test.visualize_downsampled_points.format == 'png':
                saved_path = f'{cat_path}/{category}{i - i_saved}_layer{layer}_{math.floor(iou * 1e5)}.png'
                rst_vertex = np.array(rst_tmp)
                ds_vertex = np.array(ds_tmp)
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.set_xlim3d(-view_range, view_range)
                ax.set_ylim3d(-view_range, view_range)
                ax.set_zlim3d(-view_range, view_range)
                ax.scatter(rst_vertex[:, 0], rst_vertex[:, 2], rst_vertex[:, 1], c=rst_vertex[:, 3:] / 255, marker='o',
                           s=1)
                ax.scatter(ds_vertex[:, 0], ds_vertex[:, 2], ds_vertex[:, 1], c=ds_vertex[:, 3:] / 255, marker='o', s=1)
                plt.axis('off')
                plt.grid('off')
                plt.savefig(saved_path, bbox_inches='tight')
                plt.close(fig)
            else:
                raise ValueError(
                    f'format should be png or ply, but got {config.test.visualize_downsampled_points.format}')
        pbar.update(i)
    print(f'Done! All files are saved in {base_path}')


def visualize_modelnet_downsampled_points(config, samples, index, cls_labels, mode, artifacts_path):
    base_path = f'{artifacts_path}/vis_ds_points/{mode}'
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    # get category name
    categories = []
    for cat_id in cls_labels:
        categories.append(config.datasets.mapping[int(cat_id)])
    # select samples
    samples_tmp = []
    categories_tmp = []
    if config.edgeconv_block.enable:
        idx_tmp = [[] for _ in range(len(config.edgeconv_with_ds_block.samble_downsample.M))]
    if config.neighbor2point_block.enable:
        idx_tmp = [[] for _ in range(len(config.neighbor2point_block.samble_downsample.M))]
    if config.point2point_block.enable:
        idx_tmp = [[] for _ in range(len(config.point2point_block.samble_downsample.M))]
    i_categories = []
    for cat_id in config.test.visualize_downsampled_points.vis_which:
        samples_toappend = samples[cls_labels == cat_id][:config.test.visualize_downsampled_points.num_vis]
        i_categories.append(samples_toappend.shape[0])
        samples_tmp.append(samples_toappend)
        categories_tmp.append(
            np.asarray(categories)[cls_labels == cat_id][:config.test.visualize_downsampled_points.num_vis])
        for layer, idx in enumerate(index):
            idx_tmp[layer].append(idx[cls_labels == cat_id][:config.test.visualize_downsampled_points.num_vis])
    samples = np.concatenate(samples_tmp)
    categories = np.concatenate(categories_tmp)
    index = []
    for each in idx_tmp:
        index.append(np.concatenate(each))
    # start visualization
    pbar = pkbar.Pbar(name='Generating visualized downsampled points files, please wait...', target=len(samples))
    i_saved = 0
    for i, (sample, category) in enumerate(zip(samples, categories)):

        # make every category name start from 0
        for num_category in i_categories:
            if i_saved + num_category > i:
                break
            i_saved += num_category

        xyzRGB = []
        for xyz in sample:
            xyzRGB_tmp = []
            xyzRGB_tmp.extend(list(xyz))
            xyzRGB_tmp.extend([192, 192, 192])  # gray color
            xyzRGB.append(xyzRGB_tmp)
        cat_path = f'{base_path}/{category}'
        if not os.path.exists(cat_path):
            os.makedirs(cat_path)
        for layer, idx in enumerate(index):  # idx.shape == (B, H, K)
            if layer != 0:
                idx = idx[i, 0]  # only visualize the first head
                for j in range(1, layer + 1):
                    idx = index[layer - j][i, 0][idx]  # index mapping
            else:
                idx = index[layer][i, 0]
            xyzRGB_tmp = deepcopy(xyzRGB)
            ds_tmp = []
            rst_tmp = []
            for each_idx in idx:
                xyzRGB_tmp[each_idx][3:] = [255, 0, 0]  # red color
                ds_tmp.append(xyzRGB_tmp[each_idx])
            for each in xyzRGB_tmp:
                if each not in ds_tmp:
                    rst_tmp.append(each)
            if config.test.visualize_downsampled_points.format == 'ply':
                saved_path = f'{cat_path}/{category}{i - i_saved}_layer{layer}.ply'
                vertex = PlyElement.describe(np.array(xyzRGB_tmp,
                                                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'),
                                                             ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                PlyData([vertex]).write(saved_path)
            elif config.test.visualize_downsampled_points.format == 'png':
                saved_path = f'{cat_path}/{category}{i - i_saved}_layer{layer}.png'
                rst_vertex = np.array(rst_tmp)
                ds_vertex = np.array(ds_tmp)
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.set_xlim3d(-0.6, 0.6)
                ax.set_ylim3d(-0.6, 0.6)
                ax.set_zlim3d(-0.6, 0.6)
                ax.scatter(rst_vertex[:, 0], rst_vertex[:, 2], rst_vertex[:, 1], c=rst_vertex[:, 3:] / 255, marker='o',
                           s=1)
                ax.scatter(ds_vertex[:, 0], ds_vertex[:, 2], ds_vertex[:, 1], c=ds_vertex[:, 3:] / 255, marker='o', s=1)
                plt.axis('off')
                plt.grid('off')
                plt.savefig(saved_path, bbox_inches='tight')
                plt.close(fig)
            else:
                raise ValueError(
                    f'format should be png or ply, but got {config.test.visualize_downsampled_points.format}')
        pbar.update(i)
    print(f'Done! All files are saved in {base_path}')


def visualize_modelnet_heatmap(config, samples, attention_map, cls_labels, artifacts_path):
    # this function only generates heatmap for the first downsample layer
    my_cmap = cm.get_cmap('viridis_r', samples.shape[1])
    base_path = f'{artifacts_path}/vis_heatmap/one'
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    # get category name
    categories = []
    for cat_id in cls_labels:
        categories.append(config.datasets.mapping[int(cat_id)])
    # select samples
    samples_tmp = []
    categories_tmp = []
    if config.edgeconv_block.enable:
        attention_tmp = [[] for _ in range(len(config.edgeconv_with_ds_block.samble_downsample.M))]
    if config.neighbor2point_block.enable:
        attention_tmp = [[] for _ in range(len(config.neighbor2point_block.samble_downsample.M))]
    if config.point2point_block.enable:
        attention_tmp = [[] for _ in range(len(config.point2point_block.samble_downsample.M))]
    i_categories = []
    for cat_id in config.test.visualize_attention_heatmap.vis_which:
        samples_toappend = samples[cls_labels == cat_id][:config.test.visualize_attention_heatmap.num_vis]
        i_categories.append(samples_toappend.shape[0])
        samples_tmp.append(samples_toappend)
        categories_tmp.append(
            np.asarray(categories)[cls_labels == cat_id][:config.test.visualize_attention_heatmap.num_vis])
        for layer, atten in enumerate(attention_map):
            attention_tmp[layer].append(atten[cls_labels == cat_id][:config.test.visualize_attention_heatmap.num_vis])
    samples = np.concatenate(samples_tmp)
    categories = np.concatenate(categories_tmp)
    attention_map = []
    for each in attention_tmp:
        attention_map.append(np.concatenate(each))
    # start visualization
    pbar = pkbar.Pbar(name='Generating visualized heatmap files, please wait...', target=len(samples))

    i_saved = 0
    for i, (sample, category, atten) in enumerate(zip(samples, categories, attention_map[0])):
        # make every category name start from 0
        for num_category in i_categories:
            if i_saved + num_category > i:
                break
            i_saved += num_category
        xyzRGB = []
        atten = atten[0]
        atten = (atten - np.mean(atten)) / np.std(atten) + 0.5
        for xyz, rgb in zip(sample, atten):
            xyzRGB_tmp = []
            xyzRGB_tmp.extend(list(xyz))
            RGB = 255 * np.asarray(my_cmap(rgb))[:3]
            xyzRGB_tmp.extend(list(RGB))
            xyzRGB.append(xyzRGB_tmp)
        cat_path = f'{base_path}/{category}'
        if not os.path.exists(cat_path):
            os.makedirs(cat_path)
        if config.test.visualize_attention_heatmap.format == 'ply':
            saved_path = f'{cat_path}/{category}{i - i_saved}.ply'
            vertex = PlyElement.describe(np.array(xyzRGB, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'),
                                                                 ('green', 'u1'), ('blue', 'u1')]), 'vertex')
            PlyData([vertex]).write(saved_path)
        elif config.test.visualize_attention_heatmap.format == 'png':
            saved_path = f'{cat_path}/{category}{i - i_saved}.png'
            vertex = np.array(xyzRGB)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.set_xlim3d(-0.6, 0.6)
            ax.set_ylim3d(-0.6, 0.6)
            ax.set_zlim3d(-0.6, 0.6)
            ax.scatter(vertex[:, 0], vertex[:, 2], vertex[:, 1], c=vertex[:, 3:] / 255, marker='o', s=1)
            plt.axis('off')
            plt.grid('off')
            plt.savefig(saved_path, bbox_inches='tight')
            plt.close(fig)
        else:
            raise ValueError(f'format should be png or ply, but got {config.test.visualize_downsampled_points.format}')
        pbar.update(i)
    print(f'Done! All files are saved in {base_path}')


def visualize_modelnet_downsampled_points_rs_fps(config, samples, index, cls_labels, ds_type):
    base_path = f'./artifacts/{config.wandb.name}/vis_ds_points/{ds_type}'
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    # get category name
    categories = []
    for cat_id in cls_labels:
        categories.append(config.datasets.mapping[int(cat_id)])
    # select samples
    samples_tmp = []
    categories_tmp = []
    if config.edgeconv_block.enable:
        idx_tmp = [[] for _ in range(len(config.edgeconv_with_ds_block.samble_downsample.M))]
    if config.neighbor2point_block.enable:
        idx_tmp = [[] for _ in range(3)]
    if config.point2point_block.enable:
        idx_tmp = [[] for _ in range(len(config.point2point_block.samble_downsample.M))]
    i_categories = []
    for cat_id in config.test.visualize_downsampled_points.vis_which:
        samples_toappend = samples[cls_labels == cat_id][:config.test.visualize_downsampled_points.num_vis]
        i_categories.append(samples_toappend.shape[0])
        samples_tmp.append(samples_toappend)
        categories_tmp.append(
            np.asarray(categories)[cls_labels == cat_id][:config.test.visualize_downsampled_points.num_vis])
        for layer, idx in enumerate(index):
            idx_tmp[layer].append(idx[cls_labels == cat_id][:config.test.visualize_downsampled_points.num_vis])
    samples = np.concatenate(samples_tmp)
    categories = np.concatenate(categories_tmp)
    index = []
    for each in idx_tmp:
        index.append(np.concatenate(each))
    # start visualization
    pbar = pkbar.Pbar(name='Generating visualized downsampled points files, please wait...', target=len(samples))
    i_saved = 0
    for i, (sample, category) in enumerate(zip(samples, categories)):
        # make every category name start from 0
        for num_category in i_categories:
            if i_saved + num_category > i:
                break
            i_saved += num_category
        xyzRGB = []
        for xyz in sample:
            xyzRGB_tmp = []
            xyzRGB_tmp.extend(list(xyz))
            xyzRGB_tmp.extend([192, 192, 192])  # gray color
            xyzRGB.append(xyzRGB_tmp)
        cat_path = f'{base_path}/{category}'
        if not os.path.exists(cat_path):
            os.makedirs(cat_path)
        for layer, idx in enumerate(index):  # idx.shape == (B, H, K)
            idx = index[layer][i, 0]
            xyzRGB_tmp = deepcopy(xyzRGB)
            ds_tmp = []
            rst_tmp = []
            for each_idx in idx:
                xyzRGB_tmp[each_idx][3:] = [255, 0, 0]  # red color
                ds_tmp.append(xyzRGB_tmp[each_idx])
            for each in xyzRGB_tmp:
                if each not in ds_tmp:
                    rst_tmp.append(each)
            if config.test.visualize_downsampled_points.format == 'ply':
                saved_path = f'{cat_path}/{category}{i - i_saved}_layer{layer}.ply'
                vertex = PlyElement.describe(np.array(xyzRGB_tmp,
                                                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'),
                                                             ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                PlyData([vertex]).write(saved_path)
            elif config.test.visualize_downsampled_points.format == 'png':
                saved_path = f'{cat_path}/{category}{i - i_saved}_layer{layer}.png'
                rst_vertex = np.array(rst_tmp)
                ds_vertex = np.array(ds_tmp)
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.set_xlim3d(-0.6, 0.6)
                ax.set_ylim3d(-0.6, 0.6)
                ax.set_zlim3d(-0.6, 0.6)
                ax.scatter(rst_vertex[:, 0], rst_vertex[:, 2], rst_vertex[:, 1], c=rst_vertex[:, 3:] / 255, marker='o',
                           s=1)
                ax.scatter(ds_vertex[:, 0], ds_vertex[:, 2], ds_vertex[:, 1], c=ds_vertex[:, 3:] / 255, marker='o', s=1)
                plt.axis('off')
                plt.grid('off')
                plt.savefig(saved_path, bbox_inches='tight')
                plt.close(fig)
            else:
                raise ValueError(
                    f'format should be png or ply, but got {config.test.visualize_downsampled_points.format}')
        pbar.update(i)
    print(f'Done! All files are saved in {base_path}')


def visualize_modelnet_heatmap_mode(config, samples, attention_map, cls_labels, mode, artifacts_path):
    # this function only generates heatmap for the first downsample layer
    my_cmap = cm.get_cmap('viridis_r', samples.shape[1])
    base_path = f'{artifacts_path}/vis_heatmap/{mode}'
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    # get category name
    categories = []
    for cat_id in cls_labels:
        categories.append(config.datasets.mapping[int(cat_id)])
    # select samples
    samples_tmp = []
    categories_tmp = []
    if config.neighbor2point_block.enable:
        attention_tmp = [[] for _ in range(len(config.neighbor2point_block.samble_downsample.M))]
    i_categories = []
    for cat_id in config.test.visualize_attention_heatmap.vis_which:
        samples_toappend = samples[cls_labels == cat_id][:config.test.visualize_attention_heatmap.num_vis]
        i_categories.append(samples_toappend.shape[0])
        samples_tmp.append(samples_toappend)
        categories_tmp.append(
            np.asarray(categories)[cls_labels == cat_id][:config.test.visualize_attention_heatmap.num_vis])
        for layer, atten in enumerate(attention_map):
            attention_tmp[layer].append(atten[cls_labels == cat_id][:config.test.visualize_attention_heatmap.num_vis])
    samples = np.concatenate(samples_tmp)
    categories = np.concatenate(categories_tmp)
    attention_map = []
    for each in attention_tmp:
        attention_map.append(np.concatenate(each))
    # start visualization
    pbar = pkbar.Pbar(name='Generating visualized heatmap files, please wait...', target=len(samples))

    i_saved = 0
    for i, (sample, category, atten) in enumerate(zip(samples, categories, attention_map[0])):
        # make every category name start from 0
        for num_category in i_categories:
            if i_saved + num_category > i:
                break
            i_saved += num_category
        xyzRGB = []
        atten = atten[0]
        if mode == "trained_boltzmann":
            atten = np.log(atten)
            atten = atten - np.mean(atten) + 0.5
        else:
            atten = (atten - np.mean(atten)) / np.std(atten) + 0.5
        for xyz, rgb in zip(sample, atten):
            xyzRGB_tmp = []
            xyzRGB_tmp.extend(list(xyz))
            RGB = 255 * np.asarray(my_cmap(rgb))[:3]
            xyzRGB_tmp.extend(list(RGB))
            xyzRGB.append(xyzRGB_tmp)
        cat_path = f'{base_path}/{category}'
        if not os.path.exists(cat_path):
            os.makedirs(cat_path)
        if config.test.visualize_attention_heatmap.format == 'ply':
            saved_path = f'{cat_path}/{category}{i - i_saved}.ply'
            vertex = PlyElement.describe(np.array(xyzRGB, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'),
                                                                 ('green', 'u1'), ('blue', 'u1')]), 'vertex')
            PlyData([vertex]).write(saved_path)
        elif config.test.visualize_attention_heatmap.format == 'png':
            saved_path = f'{cat_path}/{category}{i - i_saved}.png'
            vertex = np.array(xyzRGB)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.set_xlim3d(-0.6, 0.6)
            ax.set_ylim3d(-0.6, 0.6)
            ax.set_zlim3d(-0.6, 0.6)
            ax.scatter(vertex[:, 0], vertex[:, 2], vertex[:, 1], c=vertex[:, 3:] / 255, marker='o', s=1)
            plt.axis('off')
            plt.grid('off')
            plt.savefig(saved_path, bbox_inches='tight')
            plt.close(fig)
        else:
            raise ValueError(f'format should be png or ply, but got {config.test.visualize_downsampled_points.format}')
        pbar.update(i)
    print(f'Done! All files are saved in {base_path}')


def visualize_modelnet_heatmap_compare(config, samples, heatmap_dict, cls_labels, artifacts_path):
    # this function only generates heatmap for the first downsample layer
    my_cmap = cm.get_cmap('viridis_r', samples.shape[1])
    base_path = f'{artifacts_path}/vis_heatmap/compare'
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    # get category name
    categories = []
    for cat_id in cls_labels:
        categories.append(config.datasets.mapping[int(cat_id)])
    # select samples
    samples_tmp = []
    categories_tmp = []
    attention_tmp_dict = {}
    if config.neighbor2point_block.enable:
        for mode in heatmap_dict.keys():
            attention_tmp = [[] for _ in range(len(config.neighbor2point_block.samble_downsample.M))]
            attention_tmp_dict[mode] = attention_tmp
    i_categories = []
    for cat_id in config.test.visualize_attention_heatmap.vis_which:
        samples_toappend = samples[cls_labels == cat_id][:config.test.visualize_attention_heatmap.num_vis]
        i_categories.append(samples_toappend.shape[0])
        samples_tmp.append(samples_toappend)
        categories_tmp.append(
            np.asarray(categories)[cls_labels == cat_id][:config.test.visualize_attention_heatmap.num_vis])
        for mode, map in heatmap_dict.items():
            for layer, atten in enumerate(map):
                attention_tmp_dict[mode][layer].append(
                    atten[cls_labels == cat_id][:config.test.visualize_attention_heatmap.num_vis])
    samples = np.concatenate(samples_tmp)
    categories = np.concatenate(categories_tmp)
    attention_map_dict = OrderedDict()
    for mode, map in attention_tmp_dict.items():
        attention_map_dict[mode] = []
        for each in map:
            attention_map_dict[mode].append(np.concatenate(each))
    # start visualization
    pbar = pkbar.Pbar(name='Generating visualized heatmap files, please wait...', target=len(samples))

    if not config.test.visualize_attention_heatmap.format == 'png':
        raise ValueError(f'format should be png, but got {config.test.visualize_downsampled_points.format}')

    i_saved = 0
    for i, (sample, category) in enumerate(zip(samples, categories)):
        # make every category name start from 0
        for num_category in i_categories:
            if i_saved + num_category > i:
                break
            i_saved += num_category

        cat_path = f'{base_path}/{category}'
        if not os.path.exists(cat_path):
            os.makedirs(cat_path)
        saved_path = f'{cat_path}/{category}{i - i_saved}.png'
        fig = plt.figure()
        for i_plt, (mode, map_mode) in enumerate(attention_map_dict.items()):
            atten = map_mode[0][i]
            xyzRGB = []
            atten = atten[0]
            atten = (atten - np.mean(atten)) / np.std(atten) + 0.5  # normalization
            for xyz, rgb in zip(sample, atten):
                xyzRGB_tmp = []
                xyzRGB_tmp.extend(list(xyz))
                RGB = 255 * np.asarray(my_cmap(rgb))[:3]
                xyzRGB_tmp.extend(list(RGB))
                xyzRGB.append(xyzRGB_tmp)
            vertex = np.array(xyzRGB)
            ax = fig.add_subplot(2, 3, i_plt + 1, projection='3d')
            ax.set_xlim3d(-0.6, 0.6)
            ax.set_ylim3d(-0.6, 0.6)
            ax.set_zlim3d(-0.6, 0.6)
            sc = ax.scatter(vertex[:, 0], vertex[:, 2], vertex[:, 1], c=vertex[:, 3:] / 255, marker='o', s=1)
            ax.set_title(f'{mode}')
            plt.axis('off')
            plt.grid('off')
        plt.tight_layout()

        # cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.03])  # 调整位置和大小以适合图形
        # cbar = plt.colorbar(sc, cax=cbar_ax, orientation='horizontal')
        # cbar.set_label('Color Scale')
        # plt.tight_layout()
        plt.savefig(saved_path, bbox_inches='tight')
        plt.close(fig)
        pbar.update(i)
    print(f'Done! All files are saved in {base_path}')


def visualize_modelnet_combine(config, artifacts_path):
    base_path = artifacts_path
    ds_points_path = os.path.join(base_path, 'vis_ds_points')
    heatmap_path = os.path.join(base_path, 'vis_heatmap')
    to_combine_paths = {
        'heatmap': heatmap_path,
        'ds_points': ds_points_path
    }
    save_base_path = os.path.join(base_path, 'vis_compare')

    attention_modes = os.listdir(ds_points_path)
    attention_modes.sort()
    categories = os.listdir(os.path.join(ds_points_path, attention_modes[0]))
    categories.sort()

    categories_dict = {}
    num_images = 0
    for category in categories:
        cat_hm_images = os.listdir(os.path.join(heatmap_path, attention_modes[0], category))
        num_images += len(cat_hm_images)
        categories_dict[category] = cat_hm_images

    pbar = pkbar.Pbar(name='Generating visualized combined heatmap and downsampled points files, please wait...',
                      target=num_images)

    i_pbar = 0
    for i, (cat, hm_imgs) in enumerate(categories_dict.items()):
        save_cat_path = os.path.join(save_base_path, cat)
        if not os.path.exists(save_cat_path):
            os.makedirs(save_cat_path)

        for hm_img in hm_imgs:
            img_paths = []
            for attention_mode in attention_modes:
                img_paths.append(os.path.join(heatmap_path, attention_mode, cat, hm_img))
            for attention_mode in attention_modes:
                img_paths.append(
                    os.path.join(ds_points_path, attention_mode, cat, hm_img.replace('.png', '_layer0.png')))

            fig, axes = plt.subplots(nrows=len(to_combine_paths), ncols=len(attention_modes),
                                     figsize=(len(attention_modes) * 4, len(to_combine_paths) * 4))
            save_path = os.path.join(save_cat_path, hm_img)
            for i_ax, ax in enumerate(axes.flat):
                image = plt.imread(img_paths[i_ax])
                ax.imshow(image)
                ax.axis('off')
                if i_ax < len(attention_modes):
                    ax.set_title(f'{attention_modes[i_ax]}')
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()

            pbar.update(i_pbar)
            i_pbar += 1
    print(f'Done! All files are saved in {save_base_path}')


def visualize_modelnet_combine_prof(dir_name, asm="dot"):
    base_path = f'./artifacts/{dir_name}'
    heatmap_path = os.path.join(base_path, asm)
    # to_combine_paths = {
    #     'heatmap': heatmap_path,
    # }
    save_path = os.path.join(heatmap_path, f'vis_heatmap_compare_{asm}.png')

    idx_modes = os.listdir(heatmap_path)
    idx_modes.sort()
    selected_images = os.listdir(os.path.join(heatmap_path, idx_modes[0]))
    selected_images.sort()

    fig, axes = plt.subplots(nrows=len(selected_images), ncols=len(idx_modes),
                             figsize=(len(idx_modes) * 4, len(selected_images) * 4))

    img_paths = []
    for selected_image in selected_images:
        for idx_mode in idx_modes:
            img_paths.append(os.path.join(heatmap_path, idx_mode, selected_image))

    for i_ax, ax in enumerate(axes.flat):
        image = plt.imread(img_paths[i_ax])
        ax.imshow(image)
        ax.axis('off')
        if i_ax < len(idx_modes):
            ax.set_title(f'{idx_modes[i_ax]}')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f'Done!')


def visualize_modelnet_downsampled_points_few_points(config, samples, index, cls_labels, mode, artifacts_path):
    M = config.test.few_points.num_points
    if M == 256:
        s_red = 2
    elif M == 128:
        s_red = 4
    elif M == 64:
        s_red = 6
    elif M == 32:
        s_red = 8
    elif M == 16:
        s_red = 12
    elif M == 8:
        s_red = 16
    elif M == 4:
        s_red = 24
    elif M == 2:
        s_red = 32
    else:
        s_red = 1
    base_path = f'{artifacts_path}/vis_ds_points/{mode}'
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    # get category name
    categories = []
    for cat_id in cls_labels:
        categories.append(config.datasets.mapping[int(cat_id)])
    # select samples
    samples_tmp = []
    categories_tmp = []
    if config.edgeconv_block.enable:
        idx_tmp = [[] for _ in range(len(config.edgeconv_with_ds_block.samble_downsample.M))]
    if config.neighbor2point_block.enable:
        idx_tmp = [[] for _ in range(len(config.neighbor2point_block.samble_downsample.M))]
    if config.point2point_block.enable:
        idx_tmp = [[] for _ in range(len(config.point2point_block.samble_downsample.M))]
    i_categories = []
    for cat_id in config.test.visualize_downsampled_points.vis_which:
        samples_toappend = samples[cls_labels == cat_id][:config.test.visualize_downsampled_points.num_vis]
        i_categories.append(samples_toappend.shape[0])
        samples_tmp.append(samples_toappend)
        categories_tmp.append(
            np.asarray(categories)[cls_labels == cat_id][:config.test.visualize_downsampled_points.num_vis])
        for layer, idx in enumerate(index):
            idx_tmp[layer].append(idx[cls_labels == cat_id][:config.test.visualize_downsampled_points.num_vis])
    samples = np.concatenate(samples_tmp)
    categories = np.concatenate(categories_tmp)
    index = []
    for each in idx_tmp:
        index.append(np.concatenate(each))
    # start visualization
    pbar = pkbar.Pbar(name='Generating visualized downsampled points files, please wait...', target=len(samples))
    i_saved = 0
    for i, (sample, category) in enumerate(zip(samples, categories)):

        # make every category name start from 0
        for num_category in i_categories:
            if i_saved + num_category > i:
                break
            i_saved += num_category

        xyzRGB = []
        for xyz in sample:
            xyzRGB_tmp = []
            xyzRGB_tmp.extend(list(xyz))
            xyzRGB_tmp.extend([192, 192, 192])  # gray color
            xyzRGB.append(xyzRGB_tmp)
        cat_path = f'{base_path}/{category}'
        if not os.path.exists(cat_path):
            os.makedirs(cat_path)
        idx = index[0][i, 0]
        xyzRGB_tmp = deepcopy(xyzRGB)
        ds_tmp = []
        rst_tmp = []
        for each_idx in idx:
            xyzRGB_tmp[each_idx][3:] = [255, 0, 0]  # red color
            ds_tmp.append(xyzRGB_tmp[each_idx])
        for each in xyzRGB_tmp:
            if each not in ds_tmp:
                rst_tmp.append(each)
        if config.test.visualize_downsampled_points.format == 'ply':
            saved_path = f'{cat_path}/{category}{i - i_saved}_layer0.ply'
            vertex = PlyElement.describe(np.array(xyzRGB_tmp,
                                                  dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'),
                                                         ('green', 'u1'), ('blue', 'u1')]), 'vertex')
            PlyData([vertex]).write(saved_path)
        elif config.test.visualize_downsampled_points.format == 'png':
            saved_path = f'{cat_path}/{category}{i - i_saved}_layer0.png'
            rst_vertex = np.array(rst_tmp)
            ds_vertex = np.array(ds_tmp)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.set_xlim3d(-0.6, 0.6)
            ax.set_ylim3d(-0.6, 0.6)
            ax.set_zlim3d(-0.6, 0.6)
            ax.scatter(rst_vertex[:, 0], rst_vertex[:, 2], rst_vertex[:, 1], c=rst_vertex[:, 3:] / 255, marker='o', s=1)
            ax.scatter(ds_vertex[:, 0], ds_vertex[:, 2], ds_vertex[:, 1], c=ds_vertex[:, 3:] / 255, marker='o', s=s_red)
            plt.axis('off')
            plt.grid('off')
            plt.savefig(saved_path, bbox_inches='tight')
            plt.close(fig)
        else:
            raise ValueError(f'format should be png or ply, but got {config.test.visualize_downsampled_points.format}')
        pbar.update(i)
    print(f'Done! All files are saved in {base_path}')


def visualize_modelnet_downsampled_points_bin(config, samples, index, bin_prob, cls_labels, mode, artifacts_path):
    base_path = f'{artifacts_path}/vis_ds_points/{mode}'
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    # get category name
    categories = []
    for cat_id in cls_labels:
        categories.append(config.datasets.mapping[int(cat_id)])
    # select samples
    samples_tmp = []
    categories_tmp = []
    if config.neighbor2point_block.enable:
        idx_tmp = [[] for _ in range(len(config.neighbor2point_block.samble_downsample.M))]
        bin_prob_tmp = [[] for _ in range(len(config.neighbor2point_block.samble_downsample.M))]
    i_categories = []
    for cat_id in config.test.visualize_downsampled_points.vis_which:
        samples_toappend = samples[cls_labels == cat_id][:config.test.visualize_downsampled_points.num_vis]
        i_categories.append(samples_toappend.shape[0])
        samples_tmp.append(samples_toappend)
        categories_tmp.append(
            np.asarray(categories)[cls_labels == cat_id][:config.test.visualize_downsampled_points.num_vis])
        for layer, (idx, bin_p) in enumerate(zip(index, bin_prob)):
            idx_tmp[layer].append(idx[cls_labels == cat_id][:config.test.visualize_downsampled_points.num_vis])
            bin_prob_tmp[layer].append(bin_p[cls_labels == cat_id][:config.test.visualize_downsampled_points.num_vis])
    samples = np.concatenate(samples_tmp)
    categories = np.concatenate(categories_tmp)
    index = []
    bin_prob = []
    for (idx_each, bin_prob_each) in zip(idx_tmp, bin_prob_tmp):
        index.append(np.concatenate(idx_each))
        bin_prob.append(np.concatenate(bin_prob_each))
        # start visualization
    pbar = pkbar.Pbar(name='Generating visualized downsampled points files, please wait...', target=len(samples))
    i_saved = 0
    for i, (sample, category) in enumerate(zip(samples, categories)):
        # make every category name start from 0
        for num_category in i_categories:
            if i_saved + num_category > i:
                break
            i_saved += num_category

        xyzRGB = []
        for xyz in sample:
            xyzRGB_tmp = []
            xyzRGB_tmp.extend(list(xyz))
            xyzRGB_tmp.extend([192, 192, 192])  # gray color
            xyzRGB.append(xyzRGB_tmp)
        cat_path = f'{base_path}/{category}'
        if not os.path.exists(cat_path):
            os.makedirs(cat_path)
        txt_path = f'{cat_path}/{category}.txt'
        for layer, idx in enumerate(index):  # idx.shape == (B, H, K)
            with open(txt_path, 'a') as f:
                row_str = f"{category}{i - i_saved}_layer{layer}:\t{', '.join([f'{num * 100:.2f}%' for num in bin_prob[layer][i, 0]])}\n"
                f.write(row_str)
            if layer != 0:
                idx = idx[i, 0]  # only visualize the first head
                for j in range(1, layer + 1):
                    idx = index[layer - j][i, 0][idx]  # index mapping
            else:
                idx = index[layer][i, 0]

            xyzRGB_tmp = deepcopy(xyzRGB)
            ds_tmp = []
            rst_tmp = []
            for each_idx in idx:
                xyzRGB_tmp[each_idx][3:] = [255, 0, 0]  # red color
                ds_tmp.append(xyzRGB_tmp[each_idx])
            for each in xyzRGB_tmp:
                if each not in ds_tmp:
                    rst_tmp.append(each)
            if config.test.visualize_downsampled_points.format == 'ply':
                saved_path = f'{cat_path}/{category}{i - i_saved}_layer{layer}.ply'
                vertex = PlyElement.describe(np.array(xyzRGB_tmp,
                                                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'),
                                                             ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                PlyData([vertex]).write(saved_path)
            elif config.test.visualize_downsampled_points.format == 'png':
                saved_path = f'{cat_path}/{category}{i - i_saved}_layer{layer}.png'
                rst_vertex = np.array(rst_tmp)
                ds_vertex = np.array(ds_tmp)
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.set_xlim3d(-0.6, 0.6)
                ax.set_ylim3d(-0.6, 0.6)
                ax.set_zlim3d(-0.6, 0.6)
                ax.scatter(rst_vertex[:, 0], rst_vertex[:, 2], rst_vertex[:, 1], c=rst_vertex[:, 3:] / 255, marker='o',
                           s=1)
                ax.scatter(ds_vertex[:, 0], ds_vertex[:, 2], ds_vertex[:, 1], c=ds_vertex[:, 3:] / 255, marker='o', s=1)
                plt.axis('off')
                plt.grid('off')
                plt.savefig(saved_path, bbox_inches='tight')
                plt.close(fig)
            else:
                raise ValueError(
                    f'format should be png or ply, but got {config.test.visualize_downsampled_points.format}')
        pbar.update(i)
    print(f'Done! All files are saved in {base_path}')


def visualize_shapenet_downsampled_points_bin(config, samples, index, bin_prob, cls_label, shape_ious, artifacts_path):
    base_path = f'{artifacts_path}/vis_ds_points'
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    # get category name
    category_id_to_hash_code_mapping = {}
    for hash_code in list(config.datasets.mapping.keys()):
        category_id_to_hash_code_mapping[str(config.datasets.mapping[hash_code]['category_id'])] = hash_code
    categories = []
    for cat_id in cls_label:
        hash_code = category_id_to_hash_code_mapping[str(cat_id)]
        categories.append(config.datasets.mapping[hash_code]['category'])
    # select samples
    samples_tmp = []
    categories_tmp = []
    ious_tmp = []
    if config.neighbor2point_block.enable:
        idx_tmp = [[] for _ in range(len(config.neighbor2point_block.samble_downsample.M))]
        bin_p_tmp = [[] for _ in range(len(config.neighbor2point_block.samble_downsample.M))]
    i_categories = []
    for cat_id in config.test.visualize_downsampled_points.vis_which:
        samples_toappend = samples[cls_label == cat_id][:config.test.visualize_downsampled_points.num_vis]
        i_categories.append(samples_toappend.shape[0])
        samples_tmp.append(samples_toappend)
        categories_tmp.append(
            np.asarray(categories)[cls_label == cat_id][:config.test.visualize_downsampled_points.num_vis])
        ious_tmp.append(np.asarray(shape_ious)[cls_label == cat_id][:config.test.visualize_downsampled_points.num_vis])
        for layer, (idx, bin_p) in enumerate(zip(index, bin_prob)):
            idx_tmp[layer].append(idx[cls_label == cat_id][:config.test.visualize_downsampled_points.num_vis])
            bin_p_tmp[layer].append(bin_p[cls_label == cat_id][:config.test.visualize_downsampled_points.num_vis])
    samples = np.concatenate(samples_tmp)
    categories = np.concatenate(categories_tmp)
    shape_ious = np.concatenate(ious_tmp)
    index = []
    bin_prob = []
    for (idx_each, bin_p_each) in zip(idx_tmp, bin_p_tmp):
        index.append(np.concatenate(idx_each))
        bin_prob.append(np.concatenate(bin_p_each))
        # start visualization
    pbar = pkbar.Pbar(name='Generating visualized downsampled points files, please wait...', target=len(samples))
    i_saved = 0
    if config.datasets.dataset_name == "shapenet_AnTao350M":
        view_range = 0.6
    elif config.datasets.dataset_name == "shapenet_Yi650M" or config.datasets.dataset_name == "shapenet_Normal":
        view_range = 0.3
    else:
        raise ValueError(f'Unknown dataset name: {config.datasets.dataset_name}')
    for i, (sample, category, iou) in enumerate(zip(samples, categories, shape_ious)):
        # make every category name start from 0
        for num_category in i_categories:
            if i_saved + num_category > i:
                break
            i_saved += num_category
        xyzRGB = []
        for xyz in sample:
            xyzRGB_tmp = []
            xyzRGB_tmp.extend(list(xyz))
            xyzRGB_tmp.extend([192, 192, 192])  # gray color
            xyzRGB.append(xyzRGB_tmp)
        cat_path = f'{base_path}/{category}'
        if not os.path.exists(cat_path):
            os.makedirs(cat_path)

        txt_path = f'{cat_path}/{category}.txt'
        for layer, idx in enumerate(index):  # idx.shape == (B, H, K)
            with open(txt_path, 'a') as f:
                row_str = f"{category}{i - i_saved}_layer{layer}:\t{', '.join([f'{num * 100:.2f}%' for num in bin_prob[layer][i, 0]])}\n"
                f.write(row_str)
            if layer != 0:
                idx = idx[i, 0]  # only visualize the first head
                for j in range(1, layer + 1):
                    idx = index[layer - j][i, 0][idx]  # index mapping
            else:
                idx = index[layer][i, 0]
            xyzRGB_tmp = deepcopy(xyzRGB)
            ds_tmp = []
            rst_tmp = []
            for each_idx in idx:
                xyzRGB_tmp[each_idx][3:] = [255, 0, 0]  # red color
                ds_tmp.append(xyzRGB_tmp[each_idx])
            for each in xyzRGB_tmp:
                if each not in ds_tmp:
                    rst_tmp.append(each)
            if config.test.visualize_downsampled_points.format == 'ply':
                saved_path = f'{cat_path}/{category}{i - i_saved}_layer{layer}_{math.floor(iou * 1e5)}.ply'
                vertex = PlyElement.describe(np.array(xyzRGB_tmp,
                                                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'),
                                                             ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                PlyData([vertex]).write(saved_path)
            elif config.test.visualize_downsampled_points.format == 'png':
                saved_path = f'{cat_path}/{category}{i - i_saved}_layer{layer}_{math.floor(iou * 1e5)}.png'
                rst_vertex = np.array(rst_tmp)
                ds_vertex = np.array(ds_tmp)
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.set_xlim3d(-view_range, view_range)
                ax.set_ylim3d(-view_range, view_range)
                ax.set_zlim3d(-view_range, view_range)
                ax.scatter(rst_vertex[:, 0], rst_vertex[:, 2], rst_vertex[:, 1], c=rst_vertex[:, 3:] / 255, marker='o',
                           s=1)
                ax.scatter(ds_vertex[:, 0], ds_vertex[:, 2], ds_vertex[:, 1], c=ds_vertex[:, 3:] / 255, marker='o', s=1)
                plt.axis('off')
                plt.grid('off')
                plt.savefig(saved_path, bbox_inches='tight')
                plt.close(fig)
            else:
                raise ValueError(
                    f'format should be png or ply, but got {config.test.visualize_downsampled_points.format}')
        pbar.update(i)
    print(f'Done! All files are saved in {base_path}')


def visualization_heatmap(data_dict=None, save_path=None, index=None, view_range=0.6, visualization_all=False):
    counter_in_categories = {}
    if data_dict is None:

        if not os.path.exists(f'{save_path}/heat_map'):
            os.makedirs(f'{save_path}/heat_map')

        filenames = os.listdir(save_path)
        for filename in tqdm(filenames):
            if 'intermediate' not in filename:
                continue
            else:
                i = int(filename.split('_')[-1].split('.')[0])

            # for i in tqdm(range(100)):
            with open(
                    f'{save_path}/intermediate_result_{i}.pkl',
                    'rb') as f:
                data_dict = pickle.load(f)

            visualization_heatmap_one_batch(counter_in_categories, data_dict, save_path, view_range, visualization_all)
    else:
        data_dict = deepcopy(data_dict)
        if not os.path.exists(f'{save_path}/heat_map'):
            os.makedirs(f'{save_path}/heat_map')

        sampling_score_batch = data_dict['sampling_score']  # (B, num_layers, H, N)
        sample_batch = data_dict['samples']  # (B,N,3)
        label_batch = data_dict['ground_truth']
        B = sample_batch.shape[0]

        config = data_dict['config']
        if config.datasets.dataset_name == "modelnet_AnTao420M":
            mapping = config.datasets.mapping
        elif config.datasets.dataset_name == 'shapenet_AnTao350M':
            mapping = [value['category'] for value in config.datasets.mapping.values()]
        else:
            raise ValueError(f'Unknown dataset name: {config.datasets.dataset_name}')

        for j in range(B):
            if 'Shapenet' in save_path:
                pass
            elif 'Modelnet' in save_path:
                if int(label_batch[j]) not in config.test.vis_which and not visualization_all:
                    continue
            else:
                raise NotImplementedError

            sampling_score = sampling_score_batch[j][0].flatten().cpu().numpy()  # (N,)
            sample = sample_batch[j].cpu().numpy()  # (N,3)
            category = mapping[int(label_batch[j])]

            if category in counter_in_categories.keys():
                counter_in_categories[category] += 1
            else:
                counter_in_categories[category] = 1

            visualization_heatmap_one_shape(counter_in_categories[category], sample, category, sampling_score,
                                            f'{save_path}/heat_map',
                                            view_range)


def visualization_heatmap_one_batch(counter_in_categories, data_dict, save_path, view_range, visualization_all):
    config = data_dict['config']
    if config.datasets.dataset_name == "modelnet_AnTao420M":
        mapping = config.datasets.mapping
    elif config.datasets.dataset_name == 'shapenet_AnTao350M' or config.datasets.dataset_name == 'shapenet_Yi650M':
        mapping = [value['category'] for value in config.datasets.mapping.values()]
    else:
        raise ValueError(f'Unknown dataset name: {config.datasets.dataset_name}')
    sampling_score_batch = data_dict['sampling_score']  # (B, num_layers, H, N)
    sample_batch = data_dict['samples']  # (B,N,3)
    label_batch = data_dict['ground_truth']
    B = sample_batch.shape[0]
    for j in range(B):
        if 'Shapenet' in save_path:
            pass
        elif 'Modelnet' in save_path:
            if int(label_batch[j]) not in config.test.vis_which and not visualization_all:
                continue
        else:
            raise NotImplementedError

        sampling_score = sampling_score_batch[j][0].flatten().cpu().numpy()  # (N,)
        sample = sample_batch[j].cpu().numpy()  # (N,3)

        category = mapping[int(label_batch[j])]

        if category in counter_in_categories.keys():
            counter_in_categories[category] += 1
        else:
            counter_in_categories[category] = 1
        if 'rank' in counter_in_categories.keys():
            id_in_counter = counter_in_categories[category] * 2 + counter_in_categories['rank'] - 1
        else:
            id_in_counter = counter_in_categories[category]

        visualization_heatmap_one_shape(id_in_counter, sample, category, sampling_score,
                                        f'{save_path}/heat_map',
                                        view_range)


def visualization_downsampled_points(data_dict=None, save_path=None, index=None, view_range=None,
                                     visualization_all=False):
    counter_in_categories = {}
    if data_dict is None:

        if not os.path.exists(f'{save_path}/downsampled_points/'):
            os.makedirs(f'{save_path}/downsampled_points/')

        filenames = os.listdir(save_path)
        for filename in tqdm(filenames):
            if 'intermediate' not in filename:
                continue
            else:
                i = int(filename.split('_')[-1].split('.')[0])

            with open(
                    f'{save_path}/intermediate_result_{i}.pkl',
                    'rb') as f:
                data_dict = pickle.load(f)

            visualization_downsampled_points_one_batch(counter_in_categories, data_dict, save_path, view_range,
                                                       visualization_all)
    else:
        data_dict = deepcopy(data_dict)
        # save_path = f'/home/team1/cwu/FuHaoWorkspace/test_results/2024_02_04_15_47_modelnet_nostd_nonuniform_newdownsampling/downsampled_points/'

        if not os.path.exists(f'{save_path}/downsampled_points/'):
            os.makedirs(f'{save_path}/downsampled_points/')

        sample_batch = data_dict['samples']  # (B,N,3)
        label_batch = data_dict['ground_truth']  # (B,)
        idx_down_batch = data_dict['idx_down']  # B * num_layers * (H,n)

        B = sample_batch.shape[0]
        num_layers = len(idx_down_batch[0])

        config = data_dict['config']
        if config.datasets.dataset_name == "modelnet_AnTao420M":
            mapping = config.datasets.mapping
        elif config.datasets.dataset_name == 'shapenet_AnTao350M':
            mapping = [value['category'] for value in config.datasets.mapping.values()]
        else:
            raise ValueError(f'Unknown dataset name: {config.datasets.dataset_name}')

        for j in range(B):
            if 'Shapenet' in save_path:
                pass
            elif 'Modelnet' in save_path:
                if int(label_batch[j]) not in config.test.vis_which and not visualization_all:
                    continue
            else:
                raise NotImplementedError

            sample = sample_batch[j].cpu().numpy()  # (N,3)
            category = mapping[int(label_batch[j])]

            idx_down = [item.flatten().cpu().numpy() for item in idx_down_batch[j]]  # num_layers * (n,)
            for k in range(num_layers):
                if k != 0:
                    idx_down[k] = idx_down[k - 1][idx_down[k]]

                xyzRGB = []

                for xyz in sample:
                    xyzRGB_tmp = []
                    xyzRGB_tmp.extend(list(xyz))
                    # print(my_cmap)
                    # print(np.asarray(my_cmap(rgb)))
                    xyzRGB_tmp.extend([192, 192, 192])  # gray color
                    xyzRGB.append(xyzRGB_tmp)

                vertex = np.array(xyzRGB)  # (N,3+3)
                vertex[idx_down[k], 3], vertex[idx_down[k], 4], vertex[idx_down[k], 5] = 255, 0, 0  # red color

                if not os.path.exists(f'{save_path}/downsampled_points/{category}/'):
                    os.makedirs(f'{save_path}/downsampled_points/{category}/')

                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.set_xlim3d(-view_range, view_range)
                ax.set_ylim3d(-view_range, view_range)
                ax.set_zlim3d(-view_range, view_range)

                ax.scatter(vertex[:, 0], vertex[:, 2], vertex[:, 1], c=vertex[:, 3:] / 255, marker='o', s=1)
                plt.axis('off')
                plt.grid('off')

                if category in counter_in_categories.keys():
                    counter_in_categories[category] += 1
                else:
                    counter_in_categories[category] = 1

                plt.savefig(
                    f'{save_path}/downsampled_points/{category}/sample{counter_in_categories[category]}_layer{k}.png',
                    bbox_inches='tight')
                plt.close(fig)

                # print(f'.png file is saved in {saved_path}')


def visualization_downsampled_points_one_batch(counter_in_categories, data_dict, save_path, view_range,
                                               visualization_all):
    data_dict = deepcopy(data_dict)

    sample_batch = data_dict['samples']  # (B,N,3)
    label_batch = data_dict['ground_truth']  # (B,)
    idx_down_batch = data_dict['idx_down']  # B * num_layers * (H,n)
    config = data_dict['config']
    if config.datasets.dataset_name == "modelnet_AnTao420M":
        mapping = config.datasets.mapping
    elif config.datasets.dataset_name == 'shapenet_AnTao350M' or config.datasets.dataset_name == 'shapenet_Yi650M':
        mapping = [value['category'] for value in config.datasets.mapping.values()]
    else:
        raise ValueError(f'Unknown dataset name: {config.datasets.dataset_name}')
    B = sample_batch.shape[0]
    num_layers = len(idx_down_batch[0])
    for j in range(B):
        if 'Shapenet' in save_path:
            pass
        elif 'Modelnet' in save_path:
            if int(label_batch[j]) not in config.test.vis_which and not visualization_all:
                continue
        else:
            raise NotImplementedError

        sample = sample_batch[j].cpu().numpy()  # (N,3)
        category = mapping[int(label_batch[j])]

        idx_down = [item.flatten().cpu().numpy() for item in idx_down_batch[j]]  # num_layers * (n,)

        if category in counter_in_categories.keys():
            counter_in_categories[category] += 1
        else:
            counter_in_categories[category] = 1
        if 'rank' in counter_in_categories.keys():
            id_in_counter = counter_in_categories[category] * 2 + counter_in_categories['rank'] - 1
        else:
            id_in_counter = counter_in_categories[category]

        for k in range(num_layers):
            if k != 0:
                idx_down[k] = idx_down[k - 1][idx_down[k]]

            xyzRGB = []

            for xyz in sample:
                xyzRGB_tmp = []
                xyzRGB_tmp.extend(list(xyz))
                # print(my_cmap)
                # print(np.asarray(my_cmap(rgb)))
                xyzRGB_tmp.extend([192, 192, 192])  # gray color
                xyzRGB.append(xyzRGB_tmp)

            vertex = np.array(xyzRGB)  # (N,3+3)
            vertex[idx_down[k], 3], vertex[idx_down[k], 4], vertex[idx_down[k], 5] = 255, 0, 0  # red color

            if not os.path.exists(f'{save_path}/downsampled_points/{category}/'):
                os.makedirs(f'{save_path}/downsampled_points/{category}/')

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.set_xlim3d(-view_range, view_range)
            ax.set_ylim3d(-view_range, view_range)
            ax.set_zlim3d(-view_range, view_range)
            ax.scatter(vertex[:, 0], vertex[:, 2], vertex[:, 1], c=vertex[:, 3:] / 255, marker='o', s=1)
            plt.axis('off')
            plt.grid('off')

            plt.savefig(
                f'{save_path}/downsampled_points/{category}/sample{id_in_counter}_layer{k}.png',
                bbox_inches='tight')
            plt.close(fig)

            # print(f'.png file is saved in {saved_path}')


def visualization_points_in_bins(data_dict=None, save_path=None, index=None, view_range=None, visualization_all=False):
    counter_in_categories = {}
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'paleturquoise', 'violet']
    # ['red', 'orange', 'yellow', 'palegreen', 'paleturquoise', 'orchid']
    # ['red', 'orange', 'yellow', 'lightgreen', 'paleturquoise', 'orchid']
    # ['red', 'orange', 'yellow', 'palegreen', 'lightcyan', 'orchid']
    # ['red', 'orange', 'yellow', 'lime', 'cyan', 'orchid']
    # ['firebrick', 'orange', 'yellow', 'lime', 'cyan', 'orchid']
    # ['red', 'orange', 'yellow', 'lime', 'lightskyblue', 'purple']
    # ['red', 'purple', 'Yellow', 'Green', 'dodgerblue', 'olive']
    colors = [[int(round(RGorB * 255)) for RGorB in matplotlib.colors.to_rgb(color)] for color in colors]

    if data_dict is None:

        if not os.path.exists(f'{save_path}/points_in_bins'):
            os.makedirs(f'{save_path}/points_in_bins')

        filenames = os.listdir(save_path)
        for filename in tqdm(filenames):
            if 'intermediate' not in filename:
                continue
            else:
                i = int(filename.split('_')[-1].split('.')[0])

            with open(
                    f'{save_path}/intermediate_result_{i}.pkl',
                    'rb') as f:
                data_dict = pickle.load(f)

            visualization_points_in_bins_one_batch(counter_in_categories, data_dict, save_path, view_range,
                                                   visualization_all)
    else:
        data_dict = deepcopy(data_dict)
        # save_path = f'/home/team1/cwu/FuHaoWorkspace/test_results/2024_02_04_15_47_modelnet_nostd_nonuniform_newdownsampling/downsampled_points/'

        if not os.path.exists(f'{save_path}/points_in_bins/'):
            os.makedirs(f'{save_path}/points_in_bins/')

        sample_batch = data_dict['samples']  # (B,N,3)
        label_batch = data_dict['ground_truth']  # (B,)
        idx_down_batch = data_dict['idx_down']  # B * num_layers * (H,n)
        idx_in_bins_batch = data_dict['idx_in_bins']
        # (B, num_layers, num_bins, H, n) or B * num_layers * num_bins * (H,n)

        B = sample_batch.shape[0]
        num_layers = len(idx_in_bins_batch[0])
        num_bins = len(idx_in_bins_batch[0][0])

        config = data_dict['config']
        if config.datasets.dataset_name == "modelnet_AnTao420M":
            mapping = config.datasets.mapping
        elif config.datasets.dataset_name == 'shapenet_AnTao350M':
            mapping = [value['category'] for value in config.datasets.mapping.values()]
        else:
            raise ValueError(f'Unknown dataset name: {config.datasets.dataset_name}')

        for j in range(B):
            if 'Shapenet' in save_path:
                pass
            elif 'Modelnet' in save_path:
                if int(label_batch[j]) not in config.test.vis_which and not visualization_all:
                    continue
            else:
                raise NotImplementedError

            sample = sample_batch[j].cpu().numpy()  # (N,3)
            category = mapping[int(label_batch[j])]

            idx_down = [item.flatten().cpu().numpy() for item in idx_down_batch[j]]  # num_layers * (n,)

            idx_in_bins = idx_in_bins_batch[j]  # num_layers * num_bins * (H,n)
            for k in range(num_layers):
                idx_in_bins[k] = [item.flatten().cpu().numpy() for item in idx_in_bins[k]]

            if category in counter_in_categories.keys():
                counter_in_categories[category] += 1
            else:
                counter_in_categories[category] = 1

            for k in range(num_layers):
                if k != 0:
                    idx_down[k] = idx_down[k - 1][idx_down[k]]

                    for l in range(num_bins):
                        idx_in_bins[k][l] = idx_down[k - 1][idx_in_bins[k][l]]

                xyzRGB = []

                for xyz in sample:
                    xyzRGB_tmp = []
                    xyzRGB_tmp.extend(list(xyz))
                    # print(my_cmap)
                    # print(np.asarray(my_cmap(rgb)))
                    xyzRGB_tmp.extend([192, 192, 192])  # gray color
                    xyzRGB.append(xyzRGB_tmp)

                vertex = np.array(xyzRGB)  # (N,3+3)

                for l in range(num_bins):
                    vertex[idx_in_bins[k][l], 3], vertex[idx_in_bins[k][l], 4], vertex[idx_in_bins[k][l], 5] = \
                        colors[l]

                if not os.path.exists(f'{save_path}/points_in_bins/{category}/'):
                    os.makedirs(f'{save_path}/points_in_bins/{category}/')

                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.set_xlim3d(-view_range, view_range)
                ax.set_ylim3d(-view_range, view_range)
                ax.set_zlim3d(-view_range, view_range)
                ax.scatter(vertex[:, 0], vertex[:, 2], vertex[:, 1], c=vertex[:, 3:] / 255, marker='o', s=1)
                plt.axis('off')
                plt.grid('off')

                plt.savefig(
                    f'{save_path}/points_in_bins/{category}/sample{counter_in_categories[category]}_layer{k}.png',
                    bbox_inches='tight')
                plt.close(fig)


def visualization_points_in_bins_one_batch(counter_in_categories, data_dict, save_path, view_range, visualization_all):
    data_dict = deepcopy(data_dict)

    sample_batch = data_dict['samples']  # (B,N,3)
    label_batch = data_dict['ground_truth']  # (B,)
    idx_down_batch = data_dict['idx_down']  # B * num_layers * (H,n)
    idx_in_bins_batch = data_dict['idx_in_bins']
    # (B, num_layers, num_bins, H, n) or B * num_layers * num_bins * (H,n)
    B = sample_batch.shape[0]
    num_layers = len(idx_in_bins_batch[0])
    num_bins = len(idx_in_bins_batch[0][0])
    config = data_dict['config']
    if config.datasets.dataset_name == "modelnet_AnTao420M":
        mapping = config.datasets.mapping
    elif config.datasets.dataset_name == 'shapenet_AnTao350M' or config.datasets.dataset_name == 'shapenet_Yi650M':
        mapping = [value['category'] for value in config.datasets.mapping.values()]
    else:
        raise ValueError(f'Unknown dataset name: {config.datasets.dataset_name}')


    if num_bins == 2:
        colors = ['red', 'yellow']
    elif num_bins == 4:
        colors = ['red', 'yellow', 'paleturquoise', 'violet']
    elif num_bins == 6:
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'paleturquoise', 'violet']
    elif num_bins == 8:
        colors = ['red', 'orange', 'yellow', 'lime', 'cyan', 'lightgreen', 'paleturquoise', 'violet']
    elif num_bins == 10:
        colors = ['red', 'orange', 'yellow', 'firebrick', 'orchid', 'lime', 'cyan', 'lightgreen',
                  'paleturquoise', 'violet']
    elif num_bins == 12:
        colors = ['red', 'orange', 'yellow', 'firebrick', 'orchid', 'lime', 'palegreen', 'lightcyan', 'cyan',
                  'lightgreen', 'paleturquoise', 'violet']
    else:
        raise NotImplementedError
    colors = [[int(round(RGorB * 255)) for RGorB in matplotlib.colors.to_rgb(color)] for color in colors]
    for j in range(B):
        if 'Shapenet' in save_path:
            pass
        elif 'Modelnet' in save_path:
            if int(label_batch[j]) not in config.test.vis_which and not visualization_all:
                continue
        else:
            raise NotImplementedError

        sample = sample_batch[j].cpu().numpy()  # (N,3)
        category = mapping[int(label_batch[j])]

        idx_down = [item.flatten().cpu().numpy() for item in idx_down_batch[j]]  # num_layers * (n,)

        idx_in_bins = idx_in_bins_batch[j]  # num_layers * num_bins * (H,n)
        for k in range(num_layers):
            idx_in_bins[k] = [item.flatten().cpu().numpy() for item in idx_in_bins[k]]

        if category in counter_in_categories.keys():
            counter_in_categories[category] += 1
        else:
            counter_in_categories[category] = 1
        if 'rank' in counter_in_categories.keys():
            id_in_counter = counter_in_categories[category] * 2 + counter_in_categories['rank'] - 1
        else:
            id_in_counter = counter_in_categories[category]

        for k in range(num_layers):
            if k != 0:
                idx_down[k] = idx_down[k - 1][idx_down[k]]

                for l in range(num_bins):
                    idx_in_bins[k][l] = idx_down[k - 1][idx_in_bins[k][l]]

            xyzRGB = []

            for xyz in sample:
                xyzRGB_tmp = []
                xyzRGB_tmp.extend(list(xyz))
                # print(my_cmap)
                # print(np.asarray(my_cmap(rgb)))
                xyzRGB_tmp.extend([192, 192, 192])  # gray color
                xyzRGB.append(xyzRGB_tmp)

            vertex = np.array(xyzRGB)  # (N,3+3)

            # colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 215, 0], [0, 255, 255], [128, 0, 128]]

            for l in range(num_bins):
                vertex[idx_in_bins[k][l], 3], vertex[idx_in_bins[k][l], 4], vertex[idx_in_bins[k][l], 5] = \
                    colors[l]

            if not os.path.exists(f'{save_path}/points_in_bins/{category}/'):
                os.makedirs(f'{save_path}/points_in_bins/{category}/')
            # blue, darkcyan, orange, lime, yellow, Red

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.set_xlim3d(-view_range, view_range)
            ax.set_ylim3d(-view_range, view_range)
            ax.set_zlim3d(-view_range, view_range)
            ax.scatter(vertex[:, 0], vertex[:, 2], vertex[:, 1], c=vertex[:, 3:] / 255, marker='o', s=1)
            plt.axis('off')
            plt.grid('off')

            plt.savefig(
                f'{save_path}/points_in_bins/{category}/sample{id_in_counter}_layer{k}.png',
                bbox_inches='tight')
            plt.close(fig)

            # print(f'.png file is saved in {saved_path}')


def visualization_gray_one_batch(counter_in_categories, data_dict, save_path):
    # ['red', 'orange', 'yellow', 'palegreen', 'paleturquoise', 'orchid']
    # ['red', 'orange', 'yellow', 'lightgreen', 'paleturquoise', 'orchid']
    # ['red', 'orange', 'yellow', 'palegreen', 'lightcyan', 'orchid']
    # ['red', 'orange', 'yellow', 'lime', 'cyan', 'orchid']
    # ['firebrick', 'orange', 'yellow', 'lime', 'cyan', 'orchid']
    # ['red', 'orange', 'yellow', 'lime', 'lightskyblue', 'purple']
    # ['red', 'purple', 'Yellow', 'Green', 'dodgerblue', 'olive']
    color = [int(round(RGorB * 255)) for RGorB in matplotlib.colors.to_rgb('gray')]

    samples = data_dict['samples']
    config = data_dict['config']
    category_ids = data_dict['ground_truth']
    view_range = 0.6

    # (B,N,3)
    B, N, D = samples.shape

    rgb = torch.zeros((N, D), device=samples.device)
    rgb[:, 0] += color[0]
    rgb[:, 1] += color[1]
    rgb[:, 2] += color[2]

    for j, (sample, category_id) in enumerate(zip(samples, category_ids)):

        category_id = int(category_id)

        xyzRGB = torch.concat((sample, rgb), dim=1).cpu()

        # mapping: {'02691156': {'category': 'airplane', 'category_id': 0, 'parts_id': [0, 1, 2, 3]}

        if config.datasets.dataset_name == "modelnet_AnTao420M":
            mapping = config.datasets.mapping
        elif config.datasets.dataset_name == 'shapenet_AnTao350M' or config.datasets.dataset_name == 'shapenet_Yi650M':
            mapping = [value['category'] for value in config.datasets.mapping.values()]
        else:
            raise ValueError(f'Unknown dataset name: {config.datasets.dataset_name}')

        category = mapping[category_id]

        if not os.path.exists(f'{save_path}/segmentation_gray/{category}/'):
            os.makedirs(f'{save_path}/segmentation_gray/{category}/')

        if category_id in counter_in_categories.keys():
            counter_in_categories[category_id] += 1
        else:
            counter_in_categories[category_id] = 1

        vertex = np.array(xyzRGB)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlim3d(-view_range, view_range)
        ax.set_ylim3d(-view_range, view_range)
        ax.set_zlim3d(-view_range, view_range)
        ax.scatter(vertex[:, 0], vertex[:, 2], vertex[:, 1], c=vertex[:, 3:] / 255, marker='o', s=1)
        plt.axis('off')
        plt.grid('off')
        plt.savefig(f'{save_path}/segmentation_gray/{category}/sample_{counter_in_categories[category_id]}_gray.png',
                    bbox_inches='tight')
        plt.close(fig)


def visualization_points_in_gray(data_dict=None, save_path=None):
    counter_in_categories = {}
    if data_dict is None:
        file_names = os.listdir(save_path)

        for file_name in tqdm(file_names):

            if '.pkl' in file_name:
                i = int(file_name.split('_')[-1].split('.')[0])

                with open(f'{save_path}/{file_name}', 'rb') as f:
                    data_dict = pickle.load(f)

                visualization_gray_one_batch(counter_in_categories, data_dict, save_path)


    else:
        visualization_gray_one_batch(counter_in_categories, data_dict, save_path)


def visualization_histogram(data_dict=None, save_path=None, index=None, visualization_all=False):
    counter_in_categories = {}
    if data_dict is None:

        # f'/home/team1/cwu/FuHaoWorkspace/test_results/2024_02_04_15_47_modelnet_nostd_nonuniform_newdownsampling/histogram/'
        if not os.path.exists(f'{save_path}/histogram'):
            os.makedirs(f'{save_path}/histogram')

        filenames = os.listdir(save_path)
        for filename in tqdm(filenames):
            if 'intermediate' not in filename:
                continue
            else:
                i = int(filename.split('_')[-1].split('.')[0])

            with open(
                    f'{save_path}/intermediate_result_{i}.pkl', 'rb') as f:
                data_dict = pickle.load(f)

            visualization_histogram_one_batch(counter_in_categories, data_dict, save_path, visualization_all)
    else:
        data_dict = deepcopy(data_dict)
        # save_path = f'/home/team1/cwu/FuHaoWorkspace/test_results/2024_02_04_15_47_modelnet_nostd_nonuniform_newdownsampling/downsampled_points/'

        if not os.path.exists(f'{save_path}/histogram'):
            os.makedirs(f'{save_path}/histogram')

        idx_in_bins_batch = data_dict['idx_in_bins']
        # (B, num_layers, num_bins, H, n) or B * num_layers * num_bins * (H,n)
        label_batch = data_dict['ground_truth']  # (B,)
        probability_of_bins_batch = data_dict['probability_of_bins'].cpu().numpy()  # (B, num_layers, num_bins)

        # (B, num_layers, num_bins)

        B, num_layers, num_bins = probability_of_bins_batch.shape

        config = data_dict['config']
        if config.datasets.dataset_name == "modelnet_AnTao420M":
            mapping = config.datasets.mapping
        elif config.datasets.dataset_name == 'shapenet_AnTao350M':
            mapping = [value['category'] for value in config.datasets.mapping.values()]
        else:
            raise ValueError(f'Unknown dataset name: {config.datasets.dataset_name}')

        for j in range(B):
            if 'Shapenet' in save_path:
                pass
            elif 'Modelnet' in save_path:
                if int(label_batch[j]) not in config.test.vis_which and not visualization_all:
                    continue
            else:
                raise NotImplementedError

            probability_of_bins = probability_of_bins_batch[j, :, :]  # (num_layers, num_bins)
            category = mapping[int(label_batch[j])]
            idx_in_bins = idx_in_bins_batch[j]  # num_layers * num_bins * (H,n)

            for k in range(num_layers):
                idx_in_bins[k] = [item.flatten().cpu().numpy() for item in idx_in_bins[k]]
                # num_layers * num_bins * (n,)

            if category in counter_in_categories.keys():
                counter_in_categories[category] += 1
            else:
                counter_in_categories[category] = 1

            for k in range(num_layers):
                bins = np.array(range(num_bins))
                num_points_in_bins = np.array([len(item) for item in idx_in_bins[k]])
                probabilities_in_bins = probability_of_bins[k, :]

                fig = plt.figure()
                ax1 = fig.add_subplot()

                # fig, ax1 = plt.subplots()

                color = 'lightsteelblue'  # [106/255,153/255,208/255]  # 'skyblue'  # 'royalblue'  # 'cornflowerblue'  # 'royalblue' ;lightsteelblue
                ax1.set_xlabel('Bin')
                ax1.set_ylabel('Number of Points in Bins')  # , color=color)
                ax1.bar(bins, num_points_in_bins, color=color)
                ax1.tick_params(axis='y')  # , labelcolor=color)

                ax2 = ax1.twinx()

                color = 'red'  # 'darkred'
                ax2.set_ylabel('Sampling Ratio in Bins')  # , color=color)
                # ax2.set_ylim([0, 100])
                # ax2.plot(bins, probabilities_in_bins * 100, marker='o',color=color)
                ax2.plot(bins, probabilities_in_bins, linewidth=5.0, marker='o', color=color)
                ax2.tick_params(axis='y')  # , labelcolor=color)

                plt.title('Number of Points and Probabilities over Bins')

                fig.tight_layout()

                if not os.path.exists(f'{save_path}/histogram/{category}/'):
                    os.makedirs(f'{save_path}/histogram/{category}/')

                # plt.axis('off')
                # plt.grid('off')
                plt.savefig(f'{save_path}/histogram/{category}/sample{counter_in_categories[category]}_layer{k}.png',
                            bbox_inches='tight')
                plt.close(fig)


def visualization_histogram_one_batch(counter_in_categories, data_dict, save_path, visualization_all):
    data_dict = deepcopy(data_dict)

    idx_in_bins_batch = data_dict['idx_in_bins']
    # (B, num_layers, num_bins, H, n) or B * num_layers * num_bins * (H,n)
    label_batch = data_dict['ground_truth']  # (B,)
    probability_of_bins_batch = data_dict['probability_of_bins']  # (B, num_layers, num_bins)
    # probability_of_bins_batch = [torch.stack(item, dim=0) for item in probability_of_bins_batch]
    # probability_of_bins_batch = torch.stack(probability_of_bins_batch, dim=0)
    # (B, num_layers, num_bins)
    config = data_dict['config']
    if config.datasets.dataset_name == "modelnet_AnTao420M":
        mapping = config.datasets.mapping
    elif config.datasets.dataset_name == 'shapenet_AnTao350M' or config.datasets.dataset_name == 'shapenet_Yi650M':
        mapping = [value['category'] for value in config.datasets.mapping.values()]
    else:
        raise ValueError(f'Unknown dataset name: {config.datasets.dataset_name}')
    B, num_layers, num_bins = probability_of_bins_batch.shape
    for j in range(B):
        if 'Shapenet' in save_path:
            pass
        elif 'Modelnet' in save_path:
            if int(label_batch[j]) not in config.test.vis_which and not visualization_all:
                continue
        else:
            raise NotImplementedError

        probability_of_bins = probability_of_bins_batch[j, :, :]  # (num_layers, num_bins)
        category = mapping[int(label_batch[j])]
        idx_in_bins = idx_in_bins_batch[j]  # num_layers * num_bins * (H,n)

        for k in range(num_layers):
            idx_in_bins[k] = [item.flatten() for item in idx_in_bins[k]]
            # num_layers * num_bins * (n,)

        if category in counter_in_categories.keys():
            counter_in_categories[category] += 1
        else:
            counter_in_categories[category] = 1
        if 'rank' in counter_in_categories.keys():
            id_in_counter = counter_in_categories[category] * 2 + counter_in_categories['rank'] - 1
        else:
            id_in_counter = counter_in_categories[category]

        for k in range(num_layers):
            bins = np.array(range(num_bins))
            num_points_in_bins = np.array([item.nelement() for item in idx_in_bins[k]])
            probabilities_in_bins = probability_of_bins[k, :]

            fig = plt.figure()
            ax1 = fig.add_subplot()

            # fig, ax1 = plt.subplots()

            color = 'lightsteelblue'  # [106/255,153/255,208/255]  # 'skyblue'  # 'royalblue'  # 'cornflowerblue'  # 'royalblue' ;lightsteelblue
            ax1.set_xlabel('Bin')
            ax1.set_ylabel('Number of Points in Bins')  # , color=color)
            ax1.bar(bins, num_points_in_bins, color=color)
            ax1.tick_params(axis='y')  # , labelcolor=color)

            ax2 = ax1.twinx()
            color = 'red'  # 'darkred'
            ax2.set_ylabel('Sampling Ratio in Bins')  # , color=color)
            # ax2.set_ylim([0, 100])
            # ax2.plot(bins, probabilities_in_bins * 100, marker='o',color=color)
            ax2.plot(bins, probabilities_in_bins.cpu().numpy(), linewidth=5.0, marker='o', color=color)
            ax2.tick_params(axis='y')  # , labelcolor=color)

            plt.title('Number of Points and Sampling Ratio over Bins')

            fig.tight_layout()

            if not os.path.exists(f'{save_path}/histogram/{category}/'):
                os.makedirs(f'{save_path}/histogram/{category}/')

            # plt.axis('off')
            # plt.grid('off')

            plt.savefig(
                f'{save_path}/histogram/{category}/sample{id_in_counter}_layer{k}.png',
                bbox_inches='tight')
            plt.close(fig)

            # print(f'.png file is saved in {saved_path}')


def get_statistic_data_all_samples(data_dict=None, save_path=None,
                                   statistic_data_all_samples=None):
    if data_dict is None:

        # f'/home/team1/cwu/FuHaoWorkspace/test_results/2024_02_04_15_47_modelnet_nostd_nonuniform_newdownsampling/histogram/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        filenames = os.listdir(save_path)
        for filename in tqdm(filenames):
            if 'intermediate' not in filename:
                continue
            else:
                i = int(filename.split('_')[-1].split('.')[0])

            with open(f'{save_path}/intermediate_result_{i}.pkl', 'rb') as f:
                data_dict = pickle.load(f)

            statistic_data_all_samples = get_statistic_data_all_samples_one_sample(data_dict,
                                                                                   statistic_data_all_samples)
        save_statical_data(data_dict, save_path, statistic_data_all_samples)


    else:
        data_dict = deepcopy(data_dict)
        # save_path = f'/home/team1/cwu/FuHaoWorkspace/test_results/2024_02_04_15_47_modelnet_nostd_nonuniform_newdownsampling/downsampled_points/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        statistic_data_all_samples = get_statistic_data_all_samples_one_sample(data_dict, statistic_data_all_samples)
        save_statical_data(data_dict, save_path, statistic_data_all_samples)

    return statistic_data_all_samples


def get_statistic_data_all_samples_one_sample(data_dict, statistic_data_all_samples):
    data_dict = deepcopy(data_dict)

    idx_in_bins_batch = data_dict['idx_in_bins']
    # (B, num_layers, num_bins, H, n) or B * num_layers * num_bins * (H,n)
    probability_of_bins_batch = data_dict['probability_of_bins']  # (B, num_layers, num_bins)
    # probability_of_bins_batch = [torch.stack(item, dim=0) for item in probability_of_bins_batch]
    # probability_of_bins_batch = torch.stack(probability_of_bins_batch, dim=0)
    # (B, num_layers, num_bins)
    B, num_layers, num_bins = probability_of_bins_batch.shape

    if statistic_data_all_samples is None:
        num_points_in_bins_allsamples = torch.zeros((num_layers, num_bins), dtype=torch.int32)
        num_selected_points_in_bins_allsamples = torch.zeros((num_layers, num_bins), dtype=torch.int32)
        num_zeros = torch.zeros((num_layers, num_bins), dtype=torch.int32)
        num_ones = torch.zeros((num_layers, num_bins), dtype=torch.int32)
        statistic_data_all_samples = {}
    else:
        num_points_in_bins_allsamples = statistic_data_all_samples['num_points_in_bins']
        num_selected_points_in_bins_allsamples = statistic_data_all_samples['num_selected_points_in_bins']
        num_zeros = statistic_data_all_samples['num_zeros']
        num_ones = statistic_data_all_samples['num_ones']

    for j in range(B):
        probability_of_bins = probability_of_bins_batch[j, :, :]  # (num_layers, num_bins)
        idx_in_bins = idx_in_bins_batch[j]  # num_layers * num_bins * (H,n)
        for l in range(num_layers):
            idx_in_bins[l] = [item.flatten().cpu().numpy() for item in idx_in_bins[l]]

        for k in range(num_layers):
            num_points_in_bins = torch.asarray([len(item) for item in idx_in_bins[k]])
            probabilities_in_bins = probability_of_bins[k, :]
            probabilities_in_bins = torch.nan_to_num(probabilities_in_bins)

            num_points_in_bins_allsamples[k, :] += num_points_in_bins
            num_selected_points_in_bins_allsamples[k, :] += \
                torch.round(probabilities_in_bins * num_points_in_bins).to(torch.int32)
            num_zeros[k, :] += (probabilities_in_bins == 0)
            num_ones[k, :] += (probabilities_in_bins == 1)

    statistic_data_all_samples['num_points_in_bins'] = num_points_in_bins_allsamples
    statistic_data_all_samples['num_selected_points_in_bins'] = num_selected_points_in_bins_allsamples
    statistic_data_all_samples['num_zeros'] = num_zeros
    statistic_data_all_samples['num_ones'] = num_ones

    # save_statical_data(data_dict, save_path, statistic_data_all_samples)

    return statistic_data_all_samples


def save_statical_data(data_dict, save_path, statistic_data_all_samples):
    B, num_layers, num_bins = data_dict['probability_of_bins'].shape
    lines_to_save = []
    for k in range(num_layers):
        lines_to_save.append(f'\nlayer{k}:')
        lines_to_save.append(f'\n\tnum relu:{statistic_data_all_samples["num_zeros"][k, :]}')
        lines_to_save.append(f'\n\tnum saturation:{statistic_data_all_samples["num_ones"][k, :]}')
    with open(f'{save_path}/relu_and_saturation.txt', 'w') as file:
        file.writelines(lines_to_save)
    for k in range(num_layers):
        bins = np.array(range(num_bins))
        probabilities_in_bins = \
            statistic_data_all_samples['num_selected_points_in_bins'][k, :] / (
                    statistic_data_all_samples['num_points_in_bins'][k, :] + 1e-8)

        fig = plt.figure()
        ax1 = fig.add_subplot()

        # fig, ax1 = plt.subplots()

        color = 'lightsteelblue'  # [106/255,153/255,208/255]  # 'skyblue'  # 'royalblue'  # 'cornflowerblue'  # 'royalblue' ;lightsteelblue
        ax1.set_xlabel('Bin')
        ax1.set_ylabel('Number of Points in Bins')  # , color=color)
        ax1.bar(bins, statistic_data_all_samples['num_points_in_bins'][k, :].cpu().numpy(), color=color)
        ax1.tick_params(axis='y')  # , labelcolor=color)

        ax2 = ax1.twinx()

        color = 'red'  # 'darkred'
        ax2.set_ylabel('Sampling Ratio in Bins')  # , color=color)
        # ax2.set_ylim([0, 100])
        # ax2.plot(bins, probabilities_in_bins * 100, marker='o',color=color)
        ax2.plot(bins, probabilities_in_bins.cpu().numpy(), linewidth=5.0, marker='o', color=color)
        ax2.tick_params(axis='y')  # , labelcolor=color)

        plt.title('Number of Points and Sampling Ratio over Bins')

        fig.tight_layout()

        # plt.axis('off')
        # plt.grid('off')
        plt.savefig(f'{save_path}/histogram_all_samples_layer{k}.png', bbox_inches='tight')
        if ':' in save_path:
            plt.savefig(f'E:/datasets/APES/test_results/boltmannT/histogram_all_samples/{save_path.split("Modelnet_Token_")[-1]}_histogram_all_samples_layer{k}.png')
        elif 'FuHaoWorkspace' in save_path:
            if 'Modelnet_Token_' in save_path:
                plt.savefig(
                    f'/home/team1/cwu/FuHaoWorkspace/test_results/{save_path.split("Modelnet_Token_")[-1].split("/")[0]}_histogram_all_samples_layer{k}.png')
            elif 'Shapenet_Token_' in save_path:
                plt.savefig(
                    f'/home/team1/cwu/FuHaoWorkspace/test_results/{save_path.split("Shapenet_Token_")[-1].split("/")[0]}_histogram_all_samples_layer{k}.png')
            else:
                raise NotImplementedError
        plt.close(fig)


def visualize_segmentation_predictions(data_dict=None, save_path=None, index=None):
    counter_in_categories = {}
    if data_dict is None:
        file_names = os.listdir(save_path)

        for file_name in tqdm(file_names):

            if '.pkl' in file_name:
                i = int(file_name.split('_')[-1].split('.')[0])

                with open(f'{save_path}/{file_name}', 'rb') as f:
                    data_dict = pickle.load(f)

                visualization_segmentation_one_batch(counter_in_categories, data_dict, i, save_path)


    else:
        visualization_segmentation_one_batch(counter_in_categories, data_dict, index, save_path)


def visualize_segmentation_predictions_downsampled(data_dict=None, save_path=None, index=None):
    counter_in_categories_1 = {}
    counter_in_categories_2 = {}
    if data_dict is None:
        file_names = os.listdir(save_path)

        for file_name in tqdm(file_names):

            if '.pkl' in file_name:
                i = int(file_name.split('_')[-1].split('.')[0])

                with open(f'{save_path}/{file_name}', 'rb') as f:
                    data_dict = pickle.load(f)

                visualization_segmentation_one_batch_downsampled(counter_in_categories_1, data_dict, i, save_path, 1)
                visualization_segmentation_one_batch_downsampled(counter_in_categories_2, data_dict, i, save_path, 2)


    else:
        visualization_segmentation_one_batch_downsampled(counter_in_categories_1, data_dict, index, save_path, 1)
        visualization_segmentation_one_batch_downsampled(counter_in_categories_2, data_dict, index, save_path, 2)


def visualization_segmentation_one_batch_downsampled(counter_in_categories, data_dict, i, save_path, layer_index):
    data_dict = deepcopy(data_dict)

    samples = data_dict['samples']  # (B,N,3)
    config = data_dict['config']
    category_ids = data_dict['ground_truth']
    preds = data_dict['seg_predictions']  # (B,N)
    seg_labels = data_dict['seg_ground_truth']  # (B,N)
    idx_down = data_dict['idx_down']  # B * num_layers * (H,N)

    if config.datasets.dataset_name == "shapenet_AnTao350M":
        view_range = 0.6
    elif config.datasets.dataset_name == "shapenet_Yi650M" or config.datasets.dataset_name == "shapenet_Normal":
        view_range = 0.3
    else:
        raise ValueError(f'Unknown dataset name: {config.datasets.dataset_name}')
    # (B,N,3)

    B, _, _ = samples.shape

    idx_down = [[idx_down[j][k].flatten() for j in range(B)] for k in range(2)]

    for j, (sample, pred, seg_gt, category_id) in enumerate(zip(samples, preds, seg_labels, category_ids)):

        category_id = int(category_id)

        if layer_index == 1:
            idx_down_one_shape = idx_down[0][j]
        elif layer_index == 2:
            idx_down_one_shape = idx_down[0][j][idx_down[1][j]]
        else:
            raise NotImplementedError

        sample = sample[idx_down_one_shape, :]  # (N,3)
        pred = pred[idx_down_one_shape]  # (N,)
        seg_gt = seg_gt[idx_down_one_shape]  # (N,)

        # mapping: {'02691156': {'category': 'airplane', 'category_id': 0, 'parts_id': [0, 1, 2, 3]}
        xyzRGB, xyzRGB_gt = set_color_for_one_shape_seg(config, pred, sample, seg_gt)

        if 'modelnet' in config.datasets.dataset_name:
            category = config.datasets.mapping[category_id]['category']
        else:
            category = list(config.datasets.mapping.values())[category_id]['category']

        if not os.path.exists(f'{save_path}/segmentation_layer{layer_index}/{category}/'):
            os.makedirs(f'{save_path}/segmentation_layer{layer_index}/{category}/')

        if category_id in counter_in_categories.keys():
            counter_in_categories[category_id] += 1
        else:
            counter_in_categories[category_id] = 1
        if 'rank' in counter_in_categories.keys():
            id_in_counter = counter_in_categories[category_id] * 2 + counter_in_categories['rank'] - 1
        else:
            id_in_counter = counter_in_categories[category_id]

        save_figure_for_one_shape_seg(
            f'{save_path}/segmentation_layer{layer_index}/{category}/sample_{id_in_counter}_GroundTruth.png',
            f'{save_path}/segmentation_layer{layer_index}/{category}/sample_{id_in_counter}_Prediction.png',
            view_range, xyzRGB, xyzRGB_gt)


def visualization_segmentation_one_batch(counter_in_categories, data_dict, i, save_path):
    data_dict = deepcopy(data_dict)

    samples = data_dict['samples']
    config = data_dict['config']
    category_ids = data_dict['ground_truth']
    preds = data_dict['seg_predictions']
    seg_labels = data_dict['seg_ground_truth']
    if config.datasets.dataset_name == "shapenet_AnTao350M":
        view_range = 0.6
    elif config.datasets.dataset_name == "shapenet_Yi650M" or config.datasets.dataset_name == "shapenet_Normal":
        view_range = 0.3
    else:
        raise ValueError(f'Unknown dataset name: {config.datasets.dataset_name}')
    # (B,N,3)
    B, _, _ = samples.shape
    for j, (sample, pred, seg_gt, category_id) in enumerate(zip(samples, preds, seg_labels, category_ids)):

        category_id = int(category_id)

        # mapping: {'02691156': {'category': 'airplane', 'category_id': 0, 'parts_id': [0, 1, 2, 3]}
        xyzRGB, xyzRGB_gt = set_color_for_one_shape_seg(config, pred, sample, seg_gt)

        if 'modelnet' in config.datasets.dataset_name:
            category = config.datasets.mapping[category_id]['category']
        else:
            category = list(config.datasets.mapping.values())[category_id]['category']

        if not os.path.exists(f'{save_path}/segmentation/{category}/'):
            os.makedirs(f'{save_path}/segmentation/{category}/')

        if category_id in counter_in_categories.keys():
            counter_in_categories[category_id] += 1
        else:
            counter_in_categories[category_id] = 1

        if 'rank' in counter_in_categories.keys():
            id_in_counter = counter_in_categories[category_id] * 2 + counter_in_categories['rank'] - 1
        else:
            id_in_counter = counter_in_categories[category_id]

        # print(
        #     f'saving: {save_path}/segmentation/{category}/sample_{id_in_counter}_GroundTruth.png')
        save_figure_for_one_shape_seg(
            f'{save_path}/segmentation/{category}/sample_{id_in_counter}_GroundTruth.png',
            f'{save_path}/segmentation/{category}/sample_{id_in_counter}_Prediction.png',
            view_range, xyzRGB, xyzRGB_gt)


def set_color_for_one_shape_seg(config, pred, sample, seg_gt):
    xyzRGB = []
    xyzRGB_gt = []
    sample = sample.cpu()
    for xyz, p, gt in zip(sample, pred, seg_gt):
        p = p.item()
        gt = gt.item()
        xyzRGB_tmp = []
        xyzRGB_gt_tmp = []
        xyzRGB_tmp.extend(list(xyz))
        xyzRGB_tmp.extend(config.datasets.cmap[str(p)])
        xyzRGB.append(tuple(xyzRGB_tmp))
        xyzRGB_gt_tmp.extend(list(xyz))
        xyzRGB_gt_tmp.extend(config.datasets.cmap[str(gt)])
        xyzRGB_gt.append(tuple(xyzRGB_gt_tmp))

    return xyzRGB, xyzRGB_gt


def save_figure_for_one_shape_seg(gt_saved_path, pred_saved_path, view_range, xyzRGB, xyzRGB_gt):
    vertex = np.array(xyzRGB)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim3d(-view_range, view_range)
    ax.set_ylim3d(-view_range, view_range)
    ax.set_zlim3d(-view_range, view_range)
    ax.scatter(vertex[:, 0], vertex[:, 2], vertex[:, 1], c=vertex[:, 3:] / 255, marker='o', s=1)
    plt.axis('off')
    plt.grid('off')
    plt.savefig(pred_saved_path, bbox_inches='tight')
    plt.close(fig)

    vertex = np.array(xyzRGB_gt)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim3d(-view_range, view_range)
    ax.set_ylim3d(-view_range, view_range)
    ax.set_zlim3d(-view_range, view_range)
    ax.scatter(vertex[:, 0], vertex[:, 2], vertex[:, 1], c=vertex[:, 3:] / 255, marker='o', s=1)
    plt.axis('off')
    plt.grid('off')
    plt.savefig(gt_saved_path, bbox_inches='tight')
    plt.close(fig)


def visualize_few_points(M, data_dict=None, save_path=None, index=None, visualization_all=False):
    counter_in_categories = {}

    if data_dict is None:
        file_names = os.listdir(save_path)

        for file_name in tqdm(file_names):

            if '.pkl' in file_name:
                i = int(file_name.split('_')[-1].split('.')[0])

                with open(f'{save_path}/{file_name}', 'rb') as f:
                    data_dict = pickle.load(f)

                visualization_few_points_one_batch(counter_in_categories, data_dict, i,
                                                   save_path, M,
                                                   visualization_all=visualization_all)
    else:
        visualization_few_points_one_batch(counter_in_categories, data_dict, index, save_path, M,
                                           visualization_all=visualization_all)


def visualization_few_points_one_batch(counter_in_categories, data_dict, index, save_path, M, visualization_all=False):
    samples = data_dict['samples']
    config = data_dict['config']
    category_ids = data_dict['ground_truth']

    bin_prob = data_dict['raw_learned_bin_prob']
    bin_prob = torch.nn.functional.relu(bin_prob)
    # (B, num_bins)
    sampling_score = data_dict['sampling_score']
    # B * num_layers * (H, N)
    sampling_score_list = []
    for sampling_score_one_batch in sampling_score:
        sampling_score_list.append(sampling_score_one_batch[0].flatten())
    sampling_score = torch.stack(sampling_score_list, dim=0)
    # (B, N)

    idx_in_bins = data_dict['idx_in_bins']
    # B * num_layers * num_bins * (H,n)

    B, N = sampling_score.shape
    _, num_bins = bin_prob.shape

    max_num_points = torch.stack(
        [torch.asarray([idx_in_bins_in_one_bin.nelement() for idx_in_bins_in_one_bin in idx_in_bins_in_one_batch[0]])
         for idx_in_bins_in_one_batch in idx_in_bins], dim=0).to(bin_prob.device)
    # (B, num_bins)
    num_chosen_points_in_bin = calculate_num_points_to_choose(bin_prob, max_num_points, M)
    # torch.Tensor(B,num_bins)

    idx_in_bins = [[idx_in_bins_in_one_bin.flatten() for idx_in_bins_in_one_bin in idx_in_bins_in_one_batch[0]]
                   for idx_in_bins_in_one_batch in idx_in_bins]
    # B * num_bins * (n,)

    binmask_sampling_score = torch.zeros((B, num_bins, N), device=sampling_score.device) - 1
    for i in range(B):
        for j in range(num_bins):
            binmask_sampling_score[i, j, idx_in_bins[i][j]] = sampling_score[i, idx_in_bins[i][j]]
    _, sorted_index = torch.sort(binmask_sampling_score, dim=2, descending=True)

    selected_index_batch = torch.empty((B, M), device=sampling_score.device, dtype=torch.int32)
    for i in range(B):
        selected_index_one_batch = []
        for j in range(num_bins):
            selected_index_one_batch.append(sorted_index[i, j, :num_chosen_points_in_bin[i, j]])
        a = torch.concat(selected_index_one_batch, dim=0)
        selected_index_batch[i, :] = a

    counter_in_categories_copy = copy.deepcopy(counter_in_categories)
    save_figure_for_one_batch(counter_in_categories, B, M, N, category_ids, config, index, samples, save_path,
                              selected_index_batch,
                              'TopKInBin', visualization_all)

    _, sorted_index = torch.sort(sampling_score, dim=1, descending=True)
    selected_index_batch = sorted_index[:, :M]
    save_figure_for_one_batch(counter_in_categories_copy, B, M, N, category_ids, config, index, samples, save_path,
                              selected_index_batch,
                              'TopM', visualization_all)


def save_figure_for_one_batch(counter_in_categories, B, M, N, category_ids, config, i, samples, save_path,
                              selected_index_batch, mode,
                              visualization_all=False):
    if 'AnTao' in config.datasets.dataset_name:
        view_range = 0.6
    elif 'Yi' in config.datasets.dataset_name or config.datasets.dataset_name == "shapenet_Normal":
        view_range = 0.3
    else:
        raise ValueError(f'Unknown dataset name: {config.datasets.dataset_name}')

    if M == 256:
        s_red = 2
    elif M == 128:
        s_red = 4
    elif M == 64:
        s_red = 6
    elif M == 32:
        s_red = 8
    elif M == 16:
        s_red = 12
    elif M == 8:
        s_red = 16
    elif M == 4:
        s_red = 24
    elif M == 2:
        s_red = 32
    else:
        s_red = 1
    RGB_gray = torch.asarray([192, 192, 192], device=samples.device)
    RGB_gray = RGB_gray.unsqueeze(0).repeat(N, 1)
    # RGBs: (N,3)
    for j, (sample, category_id, selected_index) in enumerate(zip(samples, category_ids, selected_index_batch)):
        # sample (N,3)

        category_id = int(category_id)

        if 'Shapenet' in save_path:
            pass
        elif 'Modelnet' in save_path:
            if category_id not in config.test.vis_which and not visualization_all:
                continue
        else:
            raise NotImplementedError

        if category_id in counter_in_categories.keys():
            counter_in_categories[category_id] += 1
        else:
            counter_in_categories[category_id] = 1
        if 'rank' in counter_in_categories.keys():
            id_in_counter = counter_in_categories[category_id] * 2 + counter_in_categories['rank'] - 1
        else:
            id_in_counter = counter_in_categories[category_id]

        xyzRGB = torch.concat([sample, RGB_gray], dim=1)

        mask = torch.zeros((N,), dtype=torch.bool)
        mask[selected_index] = True

        xyzRGB_selected = xyzRGB[mask]
        xyzRGB_droped = xyzRGB[~mask]

        xyzRGB_selected[:, 3:] = torch.asarray([255, 0, 0], device=samples.device)

        xyzRGB_selected = xyzRGB_selected.cpu().numpy()
        xyzRGB_droped = xyzRGB_droped.cpu().numpy()

        if 'shapenet' in config.datasets.dataset_name:
            category = list(config.datasets.mapping.values())[category_id]['category']
        else:
            category = config.datasets.mapping[category_id]

        if not os.path.exists(f'{save_path}/few_points/{M}/{category}/'):
            os.makedirs(f'{save_path}/few_points/{M}/{category}/')

        save_figure_for_one_shape(
            f'{save_path}/few_points/{M}/{category}/sample_{id_in_counter}_{mode}.png',
            view_range, s_red, xyzRGB_selected, xyzRGB_droped)


def save_figure_for_one_shape(saved_path, view_range, s_red, xyzRGB_selected, xyzRGB_droped):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim3d(-view_range, view_range)
    ax.set_ylim3d(-view_range, view_range)
    ax.set_zlim3d(-view_range, view_range)
    ax.scatter(xyzRGB_selected[:, 0], xyzRGB_selected[:, 2], xyzRGB_selected[:, 1], c=xyzRGB_selected[:, 3:] / 255,
               marker='o', s=s_red)
    ax.scatter(xyzRGB_droped[:, 0], xyzRGB_droped[:, 2], xyzRGB_droped[:, 1], c=xyzRGB_droped[:, 3:] / 255, marker='o',
               s=1)
    plt.axis('off')
    plt.grid('off')
    plt.savefig(saved_path, bbox_inches='tight')
    plt.close(fig)


def crop_image(input_image_path, output_image_path, crop_area):
    image = Image.open(input_image_path)
    cropped_image = image.crop(crop_area)
    cropped_image.save(output_image_path)


def copy_rename(source_path, category, index, destination_path):
    import shutil

    num_points_list = [8, 16, 32, 64, 128]

    for num_points in num_points_list:
        shutil.copyfile(f'{source_path}/{num_points}/{category}/sample_{index}_TopKInBin.png',
                        f'{destination_path}/sample_{index}_{num_points}_TopKInBin.png')

        shutil.copyfile(f'{source_path}/{num_points}/{category}/sample_{index}_TopM.png',
                        f'{destination_path}/sample_{index}_{num_points}_TopM.png')


def copy_and_crop(category, index, crop_area, mode='apes'):
    import shutil

    num_points_list = [8, 16, 32, 64, 128]

    if mode == 'apes':
        for num_points in num_points_list:
            source_path = 'E:/datasets/APES/test_results/2024_02_21_01_47_Modelnet_Token_Std/few_points'
            destination_path = 'D:/master/semester7/master_arbeit/ECCV/Figures/few/pictures'

            shutil.copyfile(f'{source_path}/{num_points}/{category}/sample_{index}_TopKInBin.png',
                            f'{destination_path}/sample_{index}_{num_points}_TopKInBin.png')

            image = Image.open(f'{destination_path}/sample_{index}_{num_points}_TopKInBin.png')
            cropped_image = image.crop(crop_area)
            cropped_image.save(f'{destination_path}/sample_{index}_{num_points}_TopKInBin.png')

        source_path = 'E:/datasets/APES/test_results/2024_02_21_01_47_Modelnet_Token_Std/heat_map'
        destination_path = 'D:/master/semester7/master_arbeit/ECCV/Figures/few/pictures'

        shutil.copyfile(f'{source_path}/{category}/sample{index}_layer_0.png',
                        f'{destination_path}/sample_{index}_heatmap.png')

        image = Image.open(f'{destination_path}/sample_{index}_heatmap.png')
        cropped_image = image.crop(crop_area)
        cropped_image.save(f'{destination_path}/sample_{index}_heatmap.png')

    elif mode == 'v1':
        for num_points in num_points_list:
            source_path = 'E:/for_few_fig/APESv1/cls-APESv1-local-nocloss_few'
            destination_path = 'D:/master/semester7/master_arbeit/ECCV/Figures/few/pictures'

            shutil.copyfile(
                f'{source_path}{num_points}/vis_ds_points/local_std/{category}/{category}{index}_layer0.png',
                f'{destination_path}/sample_{index}_{num_points}_V1.png')

            image = Image.open(f'{destination_path}/sample_{index}_{num_points}_V1.png')
            cropped_image = image.crop(crop_area)
            cropped_image.save(f'{destination_path}/sample_{index}_{num_points}_V1.png')

        source_path = 'E:/for_few_fig/APESv1/cls-APESv1-local-nocloss_few8'
        destination_path = 'D:/master/semester7/master_arbeit/ECCV/Figures/few/pictures'
        shutil.copyfile(f'{source_path}/vis_heatmap/local_std/{category}/{category}{index}.png',
                        f'{destination_path}/sample_{index}_V1_heatmap.png')

        image = Image.open(f'{destination_path}/sample_{index}_V1_heatmap.png')
        cropped_image = image.crop(crop_area)
        cropped_image.save(f'{destination_path}/sample_{index}_V1_heatmap.png')

    else:
        raise NotImplementedError


def figure_for_thesis():
    bins = np.array(range(6))
    dataset = np.asarray([1024 / 6, 1024 / 6, 1024 / 6, 1024 / 6, 1024 / 6, 1024 / 6])

    fig = plt.figure()
    ax1 = fig.add_subplot()

    # fig, ax1 = plt.subplots()

    color = 'lightsteelblue'  # [106/255,153/255,208/255]  # 'skyblue'  # 'royalblue'  # 'cornflowerblue'  # 'royalblue' ;lightsteelblue
    ax1.set_xlabel('Bin')
    ax1.set_ylabel('Number of Points in Bins')  # , color=color)
    ax1.bar(bins, dataset, color=color)
    ax1.tick_params(axis='y')  # , labelcolor=color)

    # ax2 = ax1.twinx()
    #
    # color = 'red'  # 'darkred'
    # ax2.set_ylabel('Sampling Ratio in Bins')  # , color=color)
    # # ax2.set_ylim([0, 100])
    # # ax2.plot(bins, probabilities_in_bins * 100, marker='o',color=color)
    # ax2.plot(bins, probabilities_in_bins.cpu().numpy(), linewidth=5.0, marker='o', color=color)
    # ax2.tick_params(axis='y')  # , labelcolor=color)

    # plt.title('Number of Points and Sampling Ratio over Bins')

    fig.tight_layout()

    # plt.axis('off')
    # plt.grid('off')
    plt.savefig(f'C:/Users/Lenovo/Desktop/histogram_all_samples.png', bbox_inches='tight')
    plt.close(fig)
