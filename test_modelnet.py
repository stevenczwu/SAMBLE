import wandb
from omegaconf import OmegaConf
from pathlib import Path
from utils import dataloader
from models import cls_model
import torch.multiprocessing as mp
import torch.distributed as dist
import sys
import hydra
import subprocess
import datetime
import socket

from utils.ops import reshape_gathered_variable, gather_variable_from_gpus
from utils.visualization import *
from utils.visualization_data_processing import *
from utils.check_config import set_config_run



@hydra.main(version_base=None, config_path="./configs", config_name="default.yaml")
def main_with_Decorators(config):
    main_without_Decorators(config)


def main_without_Decorators(config):
    # check working directory
    try:
        assert str(Path.cwd().resolve()) == str(Path(__file__).resolve().parents[0])
    except:
        exit(f'Working directory is not the same as project root. Exit.')

    # get test configurations
    if config.usr_config:
        test_config = OmegaConf.load(config.usr_config)
        config = OmegaConf.merge(config, test_config)

    # download artifacts
    if config.wandb.enable:
        wandb.login(key=config.wandb.api_key)
        api = wandb.Api()
        artifact = api.artifact(f'{config.wandb.entity}/{config.wandb.project}/{config.wandb.name}:latest')
        if config.test.suffix.enable:
            local_path = f'./artifacts/{config.wandb.name}_{config.test.suffix.remark}'
        else:
            local_path = f'./artifacts/{config.wandb.name}'
        artifact.download(root=local_path)
    else:
        raise ValueError('W&B is not enabled!')

    # overwrite the default config with previous run config
    config.mode = 'test'
    run_config = OmegaConf.load(f'{local_path}/usr_config.yaml')
    if not config.test.suffix.enable:
        config = OmegaConf.merge(config, run_config)
    else:
        OmegaConf.save(config, f'{local_path}/usr_config_test.yaml')
        print(f'Overwrite the previous run config with new run config.')
    config = set_config_run(config, "test")

    if config.datasets.dataset_name == 'modelnet_AnTao420M':
        dataloader.download_modelnet_AnTao420M(config.datasets.url, config.datasets.saved_path)
    elif config.datasets.dataset_name == 'modelnet_Alignment1024':
        dataloader.download_modelnet_Alignment1024(config.datasets.url, config.datasets.saved_path)
    else:
        raise ValueError('Not implemented!')

    time_label = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    # multiprocessing for ddp

    if torch.cuda.is_available():
        os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'  # read .h5 file using multiprocessing will raise error
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.test.ddp.which_gpu).replace(' ', '').replace('[', '').replace(
            ']', '')
        mp.spawn(test, args=(config,), nprocs=config.test.ddp.nproc_this_node, join=True)
    else:
        raise ValueError('Please use GPU for testing!')


def test(local_rank, config):
    rank = config.test.ddp.rank_starts_from + local_rank

    hostname = socket.gethostname()
    if 'iesservergpu' in hostname:
        save_dir = f'/home/ies/fu/train_output/test_results/{config.wandb.name}/'  # f'/data/users/fu/APES/test_results/{config.wandb.name}/'
    elif hostname == 'LAPTOP-MPPIHOVR':
        save_dir = f'E:/datasets/APES/test_results/boltmannT/{config.wandb.name}/'
    else:
        save_dir = f'/home/team1/cwu/FuHaoWorkspace/test_results/{config.wandb.name}/'
    if rank == 0:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    # set files path
    if config.test.suffix.enable:
        artifacts_path = f'./artifacts/{config.wandb.name}_{config.test.suffix.remark}'
        backup_path = f'/data/users/wan/artifacts/{config.wandb.name}_{config.test.suffix.remark}'
    else:
        artifacts_path = f'./artifacts/{config.wandb.name}'
        backup_path = f'/data/users/wan/artifacts/{config.wandb.name}'
    zip_file_path = f'{artifacts_path}.zip'

    # process initialization
    os.environ['MASTER_ADDR'] = str(config.test.ddp.master_addr)
    os.environ['MASTER_PORT'] = str(config.test.ddp.master_port)
    os.environ['WORLD_SIZE'] = str(config.test.ddp.world_size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    # gpu setting
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)  # which gpu is used by current process
    print(
        f'[init] pid: {os.getpid()} - global rank: {rank} - local rank: {local_rank} - cuda: {config.test.ddp.which_gpu[local_rank]}')

    # get datasets
    if config.datasets.dataset_name == 'modelnet_AnTao420M':
        # _, test_set = dataloader.get_modelnet_dataset_AnTao420M(config.datasets.saved_path,
        _, test_set = dataloader.get_modelnet_dataset_AnTao420M(config.datasets.saved_path,  # TODO
                                                                config.train.dataloader.selected_points,
                                                                config.train.dataloader.fps,
                                                                config.train.dataloader.data_augmentation.enable,
                                                                config.train.dataloader.data_augmentation.num_aug,
                                                                config.train.dataloader.data_augmentation.jitter.enable,
                                                                config.train.dataloader.data_augmentation.jitter.std,
                                                                config.train.dataloader.data_augmentation.jitter.clip,
                                                                config.train.dataloader.data_augmentation.rotate.enable,
                                                                config.train.dataloader.data_augmentation.rotate.which_axis,
                                                                config.train.dataloader.data_augmentation.rotate.angle_range,
                                                                config.train.dataloader.data_augmentation.translate.enable,
                                                                config.train.dataloader.data_augmentation.translate.x_range,
                                                                config.train.dataloader.data_augmentation.translate.y_range,
                                                                config.train.dataloader.data_augmentation.translate.z_range,
                                                                config.train.dataloader.data_augmentation.anisotropic_scale.enable,
                                                                config.train.dataloader.data_augmentation.anisotropic_scale.x_range,
                                                                config.train.dataloader.data_augmentation.anisotropic_scale.y_range,
                                                                config.train.dataloader.data_augmentation.anisotropic_scale.z_range,
                                                                config.train.dataloader.data_augmentation.anisotropic_scale.isotropic)
    elif config.datasets.dataset_name == 'modelnet_Alignment1024':
        _, test_set = dataloader.get_modelnet_dataset_Alignment1024(config.datasets.saved_path,
                                                                    config.train.dataloader.selected_points,
                                                                    config.train.dataloader.fps,
                                                                    config.train.dataloader.data_augmentation.enable,
                                                                    config.train.dataloader.data_augmentation.num_aug,
                                                                    config.train.dataloader.data_augmentation.jitter.enable,
                                                                    config.train.dataloader.data_augmentation.jitter.std,
                                                                    config.train.dataloader.data_augmentation.jitter.clip,
                                                                    config.train.dataloader.data_augmentation.rotate.enable,
                                                                    config.train.dataloader.data_augmentation.rotate.which_axis,
                                                                    config.train.dataloader.data_augmentation.rotate.angle_range,
                                                                    config.train.dataloader.data_augmentation.translate.enable,
                                                                    config.train.dataloader.data_augmentation.translate.x_range,
                                                                    config.train.dataloader.data_augmentation.translate.y_range,
                                                                    config.train.dataloader.data_augmentation.translate.z_range,
                                                                    config.train.dataloader.data_augmentation.anisotropic_scale.enable,
                                                                    config.train.dataloader.data_augmentation.anisotropic_scale.x_range,
                                                                    config.train.dataloader.data_augmentation.anisotropic_scale.y_range,
                                                                    config.train.dataloader.data_augmentation.anisotropic_scale.z_range,
                                                                    config.train.dataloader.data_augmentation.anisotropic_scale.isotropic)
    else:
        raise ValueError('Not implemented!')

    # get sampler
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)

    # get dataloader
    test_loader = torch.utils.data.DataLoader(test_set, config.test.dataloader.batch_size_per_gpu,
                                              num_workers=config.test.dataloader.num_workers, drop_last=True,
                                              prefetch_factor=config.test.dataloader.prefetch,
                                              pin_memory=config.test.dataloader.pin_memory, sampler=test_sampler)

    # get model
    my_model = cls_model.ModelNetModel(config)
    my_model.eval()
    my_model = my_model.to(device)
    my_model = torch.nn.parallel.DistributedDataParallel(my_model)
    map_location = {'cuda:0': f'cuda:{local_rank}'}

    if config.feature_learning_block.downsample.bin.dynamic_boundaries:
        state_dict = torch.load(f'{artifacts_path}/checkpoint.pt', map_location=map_location)
        my_model.load_state_dict(state_dict['model_state_dict'])

        config.feature_learning_block.downsample.bin.dynamic_boundaries = False
        config.feature_learning_block.downsample.bin.bin_boundaries = [
            bin_boundaries[0][0, 0, 0, 1:].tolist()
            for bin_boundaries in state_dict['bin_boundaries']]
    else:
        my_model.load_state_dict(torch.load(f'{artifacts_path}/checkpoint.pt', map_location=map_location))

    # get loss function
    if config.test.label_smoothing:
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean', label_smoothing=config.test.epsilon)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    # start test
    loss_list = []
    pred_list = []
    cls_label_list = []
    sample_list = []

    # sampling_score_list = []
    # idx_down_list = []
    # idx_in_bins_list = []
    # probability_of_bins_list = []

    # vis_test_gather_dict = vis_data_structure_init(config, based_config=True)
    # for idx_mode in vis_test_gather_dict["trained"].keys():
    #     vis_test_gather_dict["trained"][idx_mode] = [[] for _ in range(len(config.feature_learning_block.downsample.M))]
    # if config.test.visualize_downsampled_points.enable:
    #     for idx_mode in vis_test_gather_dict["ds_points"].keys():
    #         vis_test_gather_dict["ds_points"][idx_mode] = [[] for _ in
    #                                                        range(len(config.feature_learning_block.downsample.M))]
    # if config.test.visualize_attention_heatmap.enable:
    #     for idx_mode in vis_test_gather_dict["heatmap"].keys():
    #         vis_test_gather_dict["heatmap"][idx_mode] = [[] for _ in
    #                                                      range(len(config.feature_learning_block.downsample.M))]

    with torch.no_grad():
        if rank == 0:
            print(
                f'Print Results: {config.test.print_results} - Visualize Downsampled Points: {config.test.visualize_downsampled_points.enable} - Visualize Heatmap: {config.test.visualize_attention_heatmap.enable}')
            pbar = pkbar.Pbar(name='Start testing, please wait...', target=len(test_loader))

        if config.test.save_pkl:
            statistic_data_all_samples = None
            if rank == 0:
                if not os.path.exists(f'{save_dir}/histogram'):
                    os.makedirs(f'{save_dir}/histogram')
            counter_in_categories_visualization_histogram = {'rank': rank}
            counter_in_categories_visualization_points_in_bins = {'rank': rank}
            counter_in_categories_visualization_downsampled_points = {'rank': rank}
            counter_in_categories_visualization_heatmap = {'rank': rank}
            counter_in_categories_visualization_few_points = {
                8: {'rank': rank},
                16: {'rank': rank},
                32: {'rank': rank},
                64: {'rank': rank},
                128: {'rank': rank}
            }

        for i, (samples, cls_labels) in enumerate(test_loader):
            samples, cls_labels = samples.to(device), cls_labels.to(device)
            preds = my_model(samples)

            if config.train.aux_loss.enable:
                preds = preds[-1]
                loss = loss_fn(preds, cls_labels)
            else:
                loss = loss_fn(preds, cls_labels)

            # collect the result among all gpus
            pred_gather_list = [torch.empty_like(preds).to(device) for _ in range(config.test.ddp.nproc_this_node)]
            cls_label_gather_list = [torch.empty_like(cls_labels).to(device) for _ in
                                     range(config.test.ddp.nproc_this_node)]

            samples = samples.permute(0, 2, 1).contiguous()  # samples: (B,3,N)->(B,N,3)
            sample_gather_list = [torch.empty_like(samples).to(device) for _ in range(config.test.ddp.nproc_this_node)]

            # vis_test_gather_dict = vis_data_gather(config, my_model, device, rank, vis_test_gather_dict)
            torch.distributed.all_gather(pred_gather_list, preds)
            torch.distributed.all_gather(cls_label_gather_list, cls_labels)
            torch.distributed.all_gather(sample_gather_list, samples)
            torch.distributed.all_reduce(loss)

            if config.test.visualize_combine.enable:
                sampling_score_all_layers = []
                idx_down_all_layers = []
                idx_in_bins_all_layers = []
                k_point_to_choose_all_layers = []

                for i_layer, downsample_module in enumerate(my_model.module.block.downsample_list):
                    downsample_module.output_variable_calculatio()

                    sampling_score_all_layers.append(
                        gather_variable_from_gpus(downsample_module, 'attention_point_score',
                                                  rank, config.test.ddp.nproc_this_node, device))

                    idx_down_all_layers.append(
                        gather_variable_from_gpus(downsample_module, 'idx',
                                                  rank, config.test.ddp.nproc_this_node, device))

                    idx_in_bins_all_layers.append(
                        gather_variable_from_gpus(downsample_module, 'idx_chunks',
                                                  rank, config.test.ddp.nproc_this_node, device))
                    k_point_to_choose_all_layers.append(
                        gather_variable_from_gpus(downsample_module, 'k_point_to_choose',
                                                  rank, config.test.ddp.nproc_this_node, device))

                    bin_prob = gather_variable_from_gpus(downsample_module, 'bin_prob',
                                                         rank, config.test.ddp.nproc_this_node, device)
                    # bin_prob.shape == (B, num_bins)

                if rank == 0:
                    # sampling_score_all_layers: num_layers * (B,H,N) -> (B, num_layers, H, N)
                    sampling_score = reshape_gathered_variable(sampling_score_all_layers)
                    # idx_down_all_layers: num_layers * (B,H,M) -> (B, num_layers, H, N)
                    idx_down = reshape_gathered_variable(idx_down_all_layers)
                    # idx_in_bins_all_layers: num_layers * (B,num_bins,1,n) or num_layers * B * num_bins * (1,n) -> (B, num_layers, num_bins, H, n) or B * num_layers * num_bins * (H,n)
                    idx_in_bins = reshape_gathered_variable(idx_in_bins_all_layers)
                    # probability_of_bins_all_layers: num_layers * (B, num_bins) -> (B, num_layers, num_bins)
                    k_point_to_choose = reshape_gathered_variable(k_point_to_choose_all_layers)

                    num_batches = len(k_point_to_choose)
                    num_layers = len(k_point_to_choose[0])
                    num_bins = len(k_point_to_choose[0][0])
                    probability_of_bins = torch.empty((num_batches, num_layers, num_bins),
                                                      dtype=torch.float)
                    for i0 in range(num_batches):
                        for j0 in range(num_layers):
                            for k0 in range(num_bins):
                                probability_of_bins[i0, j0, k0] = \
                                    k_point_to_choose[i0][j0][k0] / idx_in_bins[i0][j0][k0].nelement()

                    # sampling_score_list.append(sampling_score)
                    # idx_down_list.append(idx_down)
                    # idx_in_bins_list.append(idx_in_bins)
                    # probability_of_bins_list.append(probability_of_bins)

                    data_dict = {'sampling_score': sampling_score,  # (B, num_layers, H, N)
                                 'samples': torch.concat(sample_gather_list, dim=0),  # (B,N,3)
                                 'idx_down': idx_down,  # B * num_layers * (H,N)
                                 'idx_in_bins': idx_in_bins,
                                 # (B, num_layers, num_bins, H, n) or B * num_layers * num_bins * (H,n)
                                 'probability_of_bins': probability_of_bins,
                                 # B * num_layers * (num_bins)
                                 'ground_truth': torch.argmax(torch.concat(cls_label_gather_list, dim=0), dim=1),
                                 # (B,)
                                 'predictions': torch.argmax(torch.concat(pred_gather_list, dim=0), dim=1),  # (B,)
                                 'config': config,
                                 'raw_learned_bin_prob': bin_prob
                                 }

                    if config.test.save_pkl:
                        statistic_data_all_samples = get_statistic_data_all_samples_one_sample(
                            data_dict,
                            statistic_data_all_samples)

                        visualization_histogram_one_batch(
                            counter_in_categories_visualization_histogram,
                            data_dict, save_dir, True)

                        visualization_points_in_bins_one_batch(
                            counter_in_categories_visualization_points_in_bins,
                            data_dict, save_dir, 0.6, False)

                        visualization_downsampled_points_one_batch(
                            counter_in_categories_visualization_downsampled_points,
                            data_dict, save_dir, 0.6, False)

                        visualization_heatmap_one_batch(
                            counter_in_categories_visualization_heatmap,
                            data_dict, save_dir, 0.6, False)

                        for M in [16, 8, 32, 64, 128]:
                            visualization_few_points_one_batch(
                                counter_in_categories_visualization_few_points[M],
                                data_dict, i, save_dir, M, visualization_all=False)

                        # with open(f'{save_dir}intermediate_result_{i}.pkl', 'wb') as f:
                        #     pickle.dump(data_dict, f)

                        # if 'Yi' in config.datasets.dataset_name:
                        #     view_range = 0.3
                        # elif 'AnTao' in config.datasets.dataset_name:
                        #     view_range = 0.6
                        # visualization_heatmap(mode='modelnet', data_dict=data_dict,
                        #                       save_path=f'{save_dir}heat_map', index=i, view_range=view_range)
                        # visualization_downsampled_points(mode='modelnet', data_dict=data_dict,
                        #                                  save_path=f'{save_dir}downsampled_points', index=i,
                        #                                  view_range=view_range)
                        # visualization_points_in_bins(mode='modelnet', data_dict=data_dict,
                        #                              save_path=f'{save_dir}points_in_bins', index=i,
                        #                              view_range=view_range)
                        # visualization_histogram(mode='modelnet', data_dict=data_dict,
                        #                         save_path=f'{save_dir}histogram', index=i)
                        #
                        # if i == 0:
                        #     statistic_data_all_samples = None
                        # statistic_data_all_samples = get_statistic_data_all_samples(
                        #     mode='modelnet',
                        #     data_dict=data_dict,
                        #     save_path=save_dir,
                        #     statistic_data_all_samples=statistic_data_all_samples)
                        pass

            if rank == 0:
                preds = torch.concat(pred_gather_list, dim=0)
                pred_list.append(torch.max(preds, dim=1)[1].detach().cpu().numpy())
                cls_labels = torch.concat(cls_label_gather_list, dim=0)
                cls_label_list.append(torch.max(cls_labels, dim=1)[1].detach().cpu().numpy())
                samples = torch.concat(sample_gather_list, dim=0)
                sample_list.append(samples.permute(0, 2, 1).detach().cpu().numpy())
                loss /= config.test.ddp.nproc_this_node
                loss_list.append(loss.detach().cpu().numpy())
                pbar.update(i)

                # if config.test.sampling_score_histogram.enable:
                #     if i == 0:
                #         torch_tensor_to_save_batch = None
                #
                #     if i == len(test_loader) - 1:
                #         save_dir = 'modelnet_sampling_scores.pt'
                #     else:
                #         save_dir = None
                #
                #     idx = [torch.squeeze(torch.asarray(item)).to(samples.device) for item in
                #            vis_test_gather_dict["trained"]["idx"]]
                #     attention_map = [torch.squeeze(torch.asarray(item)).to(samples.device) for item in
                #                      vis_test_gather_dict["trained"]["attention_point_score"]]
                #
                #     torch_tensor_to_save_batch = save_sampling_score(torch_tensor_to_save_batch, samples, idx,
                #                                                      attention_map,
                #                                                      save_dir)

        if rank == 0:
            if config.test.save_pkl:
                save_statical_data(data_dict, save_dir, statistic_data_all_samples)
    # if rank == 0:
    #     preds = np.concatenate(pred_list, axis=0)
    #     cls_labels = np.concatenate(cls_label_list, axis=0)
    #     samples = np.concatenate(sample_list, axis=0)
    #
    #     vis_concat_dict = vis_data_structure_init(config, based_config=True)
    #     vis_concat_dict = vis_data_concat(len(config.feature_learning_block.downsample.M), vis_concat_dict,
    #                                       vis_test_gather_dict)
    #
    #     # calculate metrics
    #     acc = metrics.calculate_accuracy(preds, cls_labels)
    #     category_acc = metrics.calculate_category_accuracy(preds, cls_labels, config.datasets.mapping)
    #     loss = sum(loss_list) / len(loss_list)
    #     if config.test.print_results:
    #         print(f'loss: {loss}')
    #         print(f'accuracy: {acc}')
    #         for category in list(category_acc.keys()):
    #             print(f'{category}: {category_acc[category]}')
    #     with open(f'{artifacts_path}/metrics.txt', 'w') as f:
    #         f.write(f'loss: {loss}\n')
    #         f.write(f'accuracy: {acc}\n')
    #         for category in list(category_acc.keys()):
    #             f.write(f'{category}: {category_acc[category]}\n')
    #         f.close()
    #
    #     # generating visualized downsampled points files
    #     if config.test.visualize_downsampled_points.enable:
    #         ds_path = f'{artifacts_path}/vis_ds_points'
    #         if os.path.exists(ds_path):
    #             shutil.rmtree(ds_path)
    #         for idx_mode in vis_test_gather_dict["ds_points"].keys():
    #             if config.test.few_points.enable:
    #                 visualize_modelnet_downsampled_points_few_points(config, samples, index, cls_labels, idx_mode,
    #                                                                  artifacts_path)
    #             else:
    #
    #                 if len(vis_test_gather_dict["ds_points"].keys()) == 1:
    #                     index = vis_concat_dict["trained"]["idx"]
    #                 else:
    #                     index = vis_concat_dict["ds_points"][idx_mode]
    #
    #                 if config.feature_learning_block.downsample.bin.enable[0]:
    #                     visualize_modelnet_downsampled_points_bin(config, samples, index,
    #                                                               vis_concat_dict["trained"]["bin_prob"], cls_labels,
    #                                                               idx_mode, artifacts_path)
    #                 else:
    #                     visualize_modelnet_downsampled_points(config, samples, index, cls_labels, idx_mode,
    #                                                           artifacts_path)
    #     # generating visualized heatmap files
    #     if config.test.visualize_attention_heatmap.enable:
    #         hm_path = f'{artifacts_path}/vis_heatmap'
    #         if os.path.exists(hm_path):
    #             shutil.rmtree(hm_path)
    #         for idx_mode in vis_test_gather_dict["heatmap"].keys():
    #             if len(vis_test_gather_dict["heatmap"].keys()) == 1:
    #                 attention_map = vis_concat_dict["trained"]["attention_point_score"]
    #             else:
    #                 attention_map = vis_concat_dict["heatmap"][idx_mode]
    #             visualize_modelnet_heatmap_mode(config, samples, attention_map, cls_labels, idx_mode, artifacts_path)
    #         if config.feature_learning_block.downsample.boltzmann.enable[0]:
    #             aps_boltz = vis_concat_dict["trained"]["aps_boltz"]
    #             visualize_modelnet_heatmap_mode(config, samples, aps_boltz, cls_labels, 'trained_boltzmann',
    #                                             artifacts_path)
    #     # if config.test.visualize_combine.enable:
    #     #     assert config.test.visualize_downsampled_points.enable or config.test.visualize_attention_heatmap.enable, "At least one of visualize_downsampled_points or visualize_attention_heatmap must be enabled."
    #     #     visualize_modelnet_combine(config, artifacts_path)
    #
    #     # storage and backup
    #     # save_backup(artifacts_path, zip_file_path, backup_path)


if __name__ == '__main__':
    num_arguments = len(sys.argv)

    if num_arguments > 1:
        main_with_Decorators()
    else:

        
        subprocess.run('nvidia-smi', shell=True, text=True, stdout=None, stderr=subprocess.PIPE)
        config = OmegaConf.load('configs/default.yaml')
        cmd_config = {
            'usr_config': 'configs/cls.yaml',
            'datasets': 'modelnet_AnTao420M',
            'wandb': {'name': '2024_04_09_13_39_Modelnet_Token_Std_boltzmann_T0102_norm_sparsesum1_1'},
            'test': {'ddp': {'which_gpu': [3]}}
        }
        config = OmegaConf.merge(config, OmegaConf.create(cmd_config))

        dataset_config = OmegaConf.load(f'configs/datasets/{config.datasets}.yaml')
        dataset_config = OmegaConf.create({'datasets': dataset_config})
        config = OmegaConf.merge(config, dataset_config)

        main_without_Decorators(config)
