import torch
import numpy as np
import os


def vis_data_name_init(config, based_config=True):
    if based_config == True:
        new = {
            vis: [] for vis in config.test.visualize_combine.vis_which
        }
    elif config.feature_learning_block.samble_downsample.ds_which == "global_carve":
        new = {
            "col_sum": [],
            "row_std": [],
            "sparse_row_sum": [],
            "sparse_row_std": [],
            "sparse_col_sum": [],
            "sparse_col_avg": [],
            "sparse_col_sqr": []
        }
    elif config.feature_learning_block.samble_downsample.ds_which == "local":
        new = {
            "local_std": []
        }
    elif config.feature_learning_block.samble_downsample.ds_which == "local_insert":
        new = {
            "local_std": [],
            "sparse_row_std": [],
            "sparse_col_sum": [],
            "sparse_col_avg": [],
            "sparse_col_sqr": []
        }
    return new


def vis_data_structure_init(config, based_config=False):
    if config.feature_learning_block.samble_downsample.bin.enable[0]:
        trained = {
            "idx": [],
            "attention_point_score": [],
            "bin_prob": []
        }
    elif config.feature_learning_block.samble_downsample.boltzmann.enable[0]:
        trained = {
            "idx": [],
            "attention_point_score": [],
            "aps_boltz": []
        }
    else:
        trained = {
            "idx": [],
            "attention_point_score": []
        }
    ds_points = vis_data_name_init(config, based_config)
    heatmap = vis_data_name_init(config, based_config)

    if config.datasets.dataset_name == "modelnet_AnTao420M" or config.datasets.dataset_name == "modelnet_Alignment1024":
        if based_config == True:
            if config.test.visualize_downsampled_points.enable and config.test.visualize_attention_heatmap.enable:
                vis_dict = {
                    "trained": trained,
                    "ds_points": ds_points,
                    "heatmap": heatmap,
                }
            else:
                if config.test.visualize_downsampled_points.enable:
                    vis_dict = {
                        "trained": trained,
                        "ds_points": ds_points,
                    }
                elif config.test.visualize_attention_heatmap.enable:
                    vis_dict = {
                        "trained": trained,
                        "heatmap": heatmap,
                    }
                else:
                    vis_dict = {
                        "trained": trained,
                    }
        else:
            vis_dict = {
                "trained": trained,
                "ds_points": ds_points,
                "heatmap": heatmap
            }
    elif config.datasets.dataset_name == "shapenet_AnTao350M" or config.datasets.dataset_name == "shapenet_Yi650M" or config.datasets.dataset_name == "shapenet_Normal":
        vis_dict = {
            "trained": trained
        }
    else:
        raise ValueError("Please check the dataset name!")
    return vis_dict


def vis_data_extract(config, model):
    vis_dict = vis_data_structure_init(config, based_config=False)

    if config.datasets.dataset_name == "shapenet_AnTao350M" or config.datasets.dataset_name == "shapenet_Yi650M" or config.datasets.dataset_name == "shapenet_Normal":
        for i in range(len(config.feature_learning_block.samble_downsample.M)):
            # trained model
            attention_map = model.module.block.downsample_list[i].attention_points
            mask = model.module.block.downsample_list[i].mask
            sparse_attention_map = model.module.block.downsample_list[i].sparse_attention_map

            vis_dict["trained"]["idx"].append(model.module.block.downsample_list[i].idx)
            vis_dict["trained"]["attention_point_score"].append(
                model.module.block.downsample_list[i].attention_point_score)
            if config.feature_learning_block.samble_downsample.bin.enable[i]:
                bin_prob = model.module.block.downsample_list[i].bin_prob
                vis_dict["trained"]["bin_prob"].append(bin_prob.unsqueeze(1))
            elif config.feature_learning_block.samble_downsample.boltzmann.enable[i]:
                aps_boltz = model.module.block.downsample_list[i].aps_boltz
                vis_dict["trained"]["aps_boltz"].append(aps_boltz)
    elif config.datasets.dataset_name == "modelnet_AnTao420M" or config.datasets.dataset_name == "modelnet_Alignment1024":
        if config.feature_learning_block.samble_downsample.ds_which == "global_carve":
            for i in range(len(config.feature_learning_block.samble_downsample.M)):
                # trained model
                attention_map = model.module.block.downsample_list[i].attention_points
                mask = model.module.block.downsample_list[i].mask
                sparse_attention_map = model.module.block.downsample_list[i].sparse_attention_map

                vis_dict["trained"]["idx"].append(model.module.block.downsample_list[i].idx)
                vis_dict["trained"]["attention_point_score"].append(
                    model.module.block.downsample_list[i].attention_point_score)
                if config.feature_learning_block.samble_downsample.bin.enable[i]:
                    bin_prob = model.module.block.downsample_list[i].bin_prob
                    vis_dict["trained"]["bin_prob"].append(bin_prob.unsqueeze(1))
                elif config.feature_learning_block.samble_downsample.boltzmann.enable[i]:
                    aps_boltz = model.module.block.downsample_list[i].aps_boltz
                    vis_dict["trained"]["aps_boltz"].append(aps_boltz)

                # for combined visualization
                col_sum = torch.sum(attention_map, dim=-2)
                row_std = torch.std(attention_map, dim=-1)

                sparse_num = torch.sum(mask, dim=-2)

                sparse_row_sum = torch.sum(sparse_attention_map, dim=-1)

                sparse_attention_map_std = sparse_attention_map.masked_select(mask != 0).view(
                    sparse_attention_map.shape[:-1] + (model.module.block.downsample_list[i].K,))
                sparse_row_std = torch.std(sparse_attention_map_std, dim=-1)

                sparse_col_sum = torch.sum(sparse_attention_map, dim=-2)
                sparse_col_avg = sparse_col_sum / sparse_num
                sparse_col_sqr = sparse_col_avg / sparse_num

                vis_dict["heatmap"]["col_sum"].append(col_sum)
                vis_dict["heatmap"]["row_std"].append(row_std)
                vis_dict["heatmap"]["sparse_row_sum"].append(sparse_row_sum)
                vis_dict["heatmap"]["sparse_row_std"].append(sparse_row_std)
                vis_dict["heatmap"]["sparse_col_sum"].append(sparse_col_sum)
                vis_dict["heatmap"]["sparse_col_avg"].append(sparse_col_avg)
                vis_dict["heatmap"]["sparse_col_sqr"].append(sparse_col_sqr)

                for mode, value in vis_dict["heatmap"].items():
                    if config.test.few_points.enable:
                        num_points = config.test.few_points.num_points
                    else:
                        num_points = config.feature_learning_block.samble_downsample.M[i]
                    idx = value[-1].topk(num_points, dim=-1)[1]
                    vis_dict["ds_points"][mode].append(idx)
        elif config.feature_learning_block.samble_downsample.ds_which == "local":
            vis_dict = vis_data_structure_init(config, based_config=True)
            for i in range(len(config.feature_learning_block.samble_downsample.M)):
                # trained model
                attention_map = model.module.block.downsample_list[i].attention_points
                vis_dict["trained"]["idx"].append(model.module.block.downsample_list[i].idx)
                vis_dict["trained"]["attention_point_score"].append(
                    model.module.block.downsample_list[i].attention_point_score)
                # vis_dict["trained"]["sparse_attention_map"].append(sparse_attention_map)

                vis_dict["heatmap"]["local_std"] = vis_dict["trained"]["attention_point_score"]

                for mode, value in vis_dict["heatmap"].items():
                    if config.test.few_points.enable:
                        num_points = config.test.few_points.num_points
                    else:
                        num_points = config.feature_learning_block.samble_downsample.M[i]
                    idx = value[-1].topk(num_points, dim=-1)[1]
                    vis_dict["ds_points"][mode].append(idx)
        elif config.feature_learning_block.samble_downsample.ds_which == "local_insert":
            for i in range(len(config.feature_learning_block.samble_downsample.M)):
                # trained model
                attention_map = model.module.block.downsample_list[i].attention_points
                mask = model.module.block.downsample_list[i].mask
                sparse_attention_map = model.module.block.downsample_list[i].sparse_attention_map

                vis_dict["trained"]["idx"].append(model.module.block.downsample_list[i].idx)
                vis_dict["trained"]["attention_point_score"].append(
                    model.module.block.downsample_list[i].attention_point_score)
                if config.feature_learning_block.samble_downsample.bin.enable[i]:
                    bin_prob = model.module.block.downsample_list[i].bin_prob
                    vis_dict["trained"]["bin_prob"].append(bin_prob.unsqueeze(1))
                elif config.feature_learning_block.samble_downsample.boltzmann.enable[i]:
                    aps_boltz = model.module.block.downsample_list[i].aps_boltz
                    vis_dict["trained"]["aps_boltz"].append(aps_boltz)

                # for combined visualization
                local_std = torch.std(attention_map, dim=-1, unbiased=False)[:, :, :, 0]

                sparse_num = torch.sum(mask, dim=-2)

                sparse_attention_map_std = sparse_attention_map.masked_select(mask != 0).view(
                    sparse_attention_map.shape[:-1] + (model.module.block.downsample_list[i].K,))
                sparse_row_std = torch.std(sparse_attention_map_std, dim=-1)

                sparse_col_sum = torch.sum(sparse_attention_map, dim=-2)
                sparse_col_avg = sparse_col_sum / sparse_num
                sparse_col_sqr = sparse_col_avg / sparse_num

                vis_dict["heatmap"]["local_std"].append(local_std)
                vis_dict["heatmap"]["sparse_row_std"].append(sparse_row_std)
                vis_dict["heatmap"]["sparse_col_sum"].append(sparse_col_sum)
                vis_dict["heatmap"]["sparse_col_avg"].append(sparse_col_avg)
                vis_dict["heatmap"]["sparse_col_sqr"].append(sparse_col_sqr)

                for mode, value in vis_dict["heatmap"].items():
                    if config.test.few_points.enable:
                        num_points = config.test.few_points.num_points
                    else:
                        num_points = config.feature_learning_block.samble_downsample.M[i]
                    idx = value[-1].topk(num_points, dim=-1)[1]
                    vis_dict["ds_points"][mode].append(idx)
        else:
            raise ValueError(
                f'Only support local_insert and global_carve downsample mode! Got: {config.datasets.dataset_name}')
    else:
        raise ValueError(f'please set the correct dataset name in config')
    return vis_dict


def vis_data_gather(config, model, device, rank, vis_test_gather_dict):
    vis_all_dict = vis_data_extract(config, model)
    # if config.test.visualize_combine.enable:
    #     idx_modes = config.test.visualize_combine.vis_which
    # else:
    #     idx_modes = [config.feature_learning_block.downsample.idx_mode[0]]
    for vis_type in vis_test_gather_dict.keys():
        for idx_mode in vis_test_gather_dict[vis_type].keys():
            gather_list = [[torch.empty_like(vis_all_dict[vis_type][idx_mode][j]).to(device) for _ in
                            range(config.test.ddp.nproc_this_node)] for j in
                           range(len(config.feature_learning_block.samble_downsample.M))]
            for j in range(len(config.feature_learning_block.samble_downsample.M)):
                torch.distributed.all_gather(gather_list[j], vis_all_dict[vis_type][idx_mode][j])
                if rank == 0:
                    index = torch.concat(gather_list[j], dim=0)
                    vis_test_gather_dict[vis_type][idx_mode][j].append(index.detach().cpu().numpy())
    return vis_test_gather_dict


def vis_data_concat(len_ds, vis_concat_dict, vis_test_gather_dict):
    for vis_type in vis_test_gather_dict.keys():
        for idx_mode in vis_test_gather_dict[vis_type].keys():
            for j in range(len_ds):
                vis_concat_dict[vis_type][idx_mode].append(
                    np.concatenate(vis_test_gather_dict[vis_type][idx_mode][j], axis=0))
    return vis_concat_dict


def save_sampling_score(torch_tensor_to_save_batch, points: torch.Tensor, idx: list[torch.Tensor],
                        attention_score: list[torch.Tensor],
                        save_dir='sampling_scores.pt') -> torch.Tensor:
    """
    save the sampling_score in a file as torch.Tensor[B,N,xyz+num_layers]
    :param points: Input tensor of the model, [B,N,3]
    :param idx: indexes of sampled points, a list of torch.Tensor
    :param attention_score: attention_score of input points of each layer, a list of torch.Tensor
    :param save_dir: direction to save the file
    :return: torch_tensor_to_save
    """

    # torch_tensor_to_save = torch.empty(points.shape[0], points.shape[1], 0)
    if len(idx[0].shape) == 3:
        idx = [item[-1, :, :] for item in idx]
        attention_score = [item[-1, :, :] for item in attention_score]

    reshaped_attention_score = None
    for i in range(len(idx) - 1, -1, -1):

        if reshaped_attention_score is None:
            reshaped_attention_score = torch.unsqueeze(attention_score[i], dim=1)
        else:

            reshaped_attention_score_new = torch.zeros(attention_score[i].shape[0], reshaped_attention_score.shape[1],
                                                       attention_score[i].shape[1]
                                                       ).to(points.device) - 2

            for b in range(attention_score[i].shape[0]):
                reshaped_attention_score_new[b, :, idx[i][b, :]] = reshaped_attention_score[b, :, :]

            reshaped_attention_score = torch.concat(
                [torch.unsqueeze(attention_score[i], dim=1), reshaped_attention_score_new], dim=1)

    torch_tensor_to_save = torch.concat([points, reshaped_attention_score], dim=1)

    if torch_tensor_to_save_batch is None:
        torch_tensor_to_save_batch = torch_tensor_to_save
    else:
        torch_tensor_to_save_batch = torch.concat(
            [torch_tensor_to_save_batch, torch_tensor_to_save], dim=0)

    if save_dir is not None:
        torch.save(torch_tensor_to_save_batch, save_dir)
        print(f'save .pt at {save_dir}')

    return torch_tensor_to_save_batch
