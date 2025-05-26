import wandb
import hydra
import subprocess
import socket
import sys
from omegaconf import OmegaConf
from pathlib import Path
import torch.multiprocessing as mp
import torch.distributed as dist

from utils import dataloader
from models import seg_model
from utils.visualization import *
from utils.visualization_data_processing import *
from utils.check_config import set_config_run

from utils.ops import reshape_gathered_variable, gather_variable_from_gpus


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

    if config.datasets.dataset_name == 'shapenet_Yi650M':
        dataloader.download_shapenet_Yi650M(config.datasets.url, config.datasets.saved_path)
    elif config.datasets.dataset_name == 'shapenet_AnTao350M':
        dataloader.download_shapenet_AnTao350M(config.datasets.url, config.datasets.saved_path)
    elif config.datasets.dataset_name == 'shapenet_Normal':
        dataloader.download_shapenet_Normal(config.datasets.url, config.datasets.saved_path)
    else:
        raise ValueError('Not implemented!')

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
        save_dir = f'/data/users/fu/APES/test_results/{config.wandb.name}/'
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
    if config.datasets.dataset_name == 'shapenet_Yi650M':
        _, _, _, test_set = dataloader.get_shapenet_dataset_Yi650M(config.datasets.saved_path, config.datasets.mapping,
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
                                                                   config.train.dataloader.data_augmentation.anisotropic_scale.isotropic,
                                                                   config.test.dataloader.vote.enable,
                                                                   config.test.dataloader.vote.num_vote)
    elif config.datasets.dataset_name == 'shapenet_AnTao350M':
        _, _, _, test_set = dataloader.get_shapenet_dataset_AnTao350M(config.datasets.saved_path,
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
                                                                      config.train.dataloader.data_augmentation.anisotropic_scale.isotropic,
                                                                      config.test.dataloader.vote.enable,
                                                                      config.test.dataloader.vote.num_vote)
    elif config.datasets.dataset_name == 'shapenet_Normal':
        _, _, _, test_set = dataloader.get_shapenet_dataset_Normal(config.datasets, config.train.dataloader,
                                                                   config.test.dataloader.vote)
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
    my_model = seg_model.ShapeNetModel(config)
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
    seg_label_list = []
    cls_label_list = []
    sample_list = []
    pred_novo_list = []
    loss_novo_list = []
    # vis_test_gather_dict = vis_data_structure_init(config, based_config=True)
    # for mode in vis_test_gather_dict["trained"].keys():
    #     vis_test_gather_dict["trained"][mode] = [[] for _ in range(len(config.feature_learning_block.downsample.M))]

    with torch.no_grad():
        if rank == 0:
            print(
                f'Print Results: {config.test.print_results} - Visualize Predictions: {config.test.visualize_preds.enable} - Visualize Downsampled Points: {config.test.visualize_downsampled_points.enable}')
            pbar = pkbar.Pbar(name='Start testing, please wait...', target=len(test_loader))

        if config.test.save_pkl:
            counter_in_categories_visualize_segmentation_predictions = {'rank': rank}
            counter_in_categories_visualize_segmentation_predictions_downsampled_1 = {'rank': rank}
            counter_in_categories_visualize_segmentation_predictions_downsampled_2 = {'rank': rank}
            statistic_data_all_samples = None
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

        for i, (samples, seg_labels, cls_label) in enumerate(test_loader):
            seg_labels, cls_label = seg_labels.to(device), cls_label.to(device)
            if config.test.dataloader.vote.enable:
                preds_list = []
                for samples_vote in samples:
                    samples_vote = samples_vote.to(device)
                    preds = my_model(samples_vote, cls_label)
                    preds_list.append(preds)
                preds = torch.mean(torch.stack(preds_list), dim=0)
                samples = samples[0].to(device)

                # collect no voting result
                preds_novo = preds_list[0]
                loss_novo = loss_fn(preds_novo, seg_labels)
                pred_novo_gather_list = [torch.empty_like(preds_novo).to(device) for _ in
                                         range(config.test.ddp.nproc_this_node)]
                torch.distributed.all_gather(pred_novo_gather_list, preds_novo)
                if rank == 0:
                    preds_novo = torch.concat(pred_novo_gather_list, dim=0)
                    pred_novo_list.append(torch.max(preds_novo.permute(0, 2, 1), dim=2)[1].detach().cpu().numpy())
                    loss_novo /= config.test.ddp.nproc_this_node
                    loss_novo_list.append(loss_novo.detach().cpu().numpy())
            else:
                samples = samples.to(device)
                preds = my_model(samples, cls_label)
            loss = loss_fn(preds, seg_labels)

            # collect the result among all gpus
            pred_gather_list = [torch.empty_like(preds).to(device) for _ in range(config.test.ddp.nproc_this_node)]
            seg_label_gather_list = [torch.empty_like(seg_labels).to(device) for _ in
                                     range(config.test.ddp.nproc_this_node)]
            cls_label_gather_list = [torch.empty_like(cls_label).to(device) for _ in
                                     range(config.test.ddp.nproc_this_node)]
            sample_gather_list = [torch.empty_like(samples).to(device) for _ in range(config.test.ddp.nproc_this_node)]

            # vis_test_gather_dict = vis_data_gather(config, my_model, device, rank, vis_test_gather_dict)

            torch.distributed.all_gather(pred_gather_list, preds)
            torch.distributed.all_gather(seg_label_gather_list, seg_labels)
            torch.distributed.all_gather(cls_label_gather_list, cls_label)
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

                if rank == 0:
                    # sampling_score_all_layers: num_layers * (B,H,N) -> (B, num_layers, H, N)
                    sampling_score = reshape_gathered_variable(sampling_score_all_layers)
                    # idx_down_all_layers: num_layers * (B,H,M) -> (B, num_layers, H, N)
                    idx_down = reshape_gathered_variable(idx_down_all_layers)
                    # idx_in_bins_all_layers: num_layers * (B,num_bins,1,n) or num_layers * B * num_bins * (1,n) -> (B, num_layers, num_bins, H, n) or B * num_layers * num_bins * (H,n)
                    idx_in_bins = reshape_gathered_variable(idx_in_bins_all_layers)
                    # k_point_to_choose_all_layers: num_layers * (B, num_bins) -> (B, num_layers, num_bins)
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
                                 'samples': torch.concat(sample_gather_list, dim=0).permute(0, 2, 1),  # (B,N,3)
                                 'idx_down': idx_down,  # B * num_layers * (H,N)
                                 'idx_in_bins': idx_in_bins,
                                 # (B, num_layers, num_bins, H, n) or B * num_layers * num_bins * (H,n)
                                 'probability_of_bins': probability_of_bins,
                                 # B * num_layers * (num_bins)
                                 'ground_truth': torch.argmax(torch.concat(cls_label_gather_list, dim=0), dim=1),
                                 # (B,)
                                 'seg_predictions': torch.argmax(torch.concat(pred_gather_list, dim=0), dim=1),  # (B,)
                                 'seg_ground_truth': torch.argmax(torch.concat(seg_label_gather_list, dim=0), dim=1),
                                 'config': config,
                                 'raw_learned_bin_prob': bin_prob
                                 }
                    # print(f'samples.shape:{torch.concat(sample_gather_list, dim=0).shape}')

                    if config.test.save_pkl:
                        visualization_segmentation_one_batch(
                            counter_in_categories_visualize_segmentation_predictions,
                            data_dict, i, save_dir)

                        visualization_segmentation_one_batch_downsampled(
                            counter_in_categories_visualize_segmentation_predictions_downsampled_1,
                            data_dict, i, save_dir, 1)
                        visualization_segmentation_one_batch_downsampled(
                            counter_in_categories_visualize_segmentation_predictions_downsampled_2,
                            data_dict, i, save_dir, 2)

                        statistic_data_all_samples = get_statistic_data_all_samples_one_sample(
                            data_dict,
                            statistic_data_all_samples)

                        visualization_histogram_one_batch(
                            counter_in_categories_visualization_histogram,
                            data_dict, save_dir, False)

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
                        # print(f'save{i}')

            if rank == 0:
                preds = torch.concat(pred_gather_list, dim=0)
                pred_list.append(torch.max(preds, dim=1)[1].detach().cpu().numpy())
                seg_labels = torch.concat(seg_label_gather_list, dim=0)
                seg_label_list.append(torch.max(seg_labels.permute(0, 2, 1), dim=2)[1].detach().cpu().numpy())
                cls_label = torch.concat(cls_label_gather_list, dim=0)
                cls_label_list.append(torch.max(cls_label[:, :, 0], dim=1)[1].detach().cpu().numpy())
                samples = torch.concat(sample_gather_list, dim=0)  # samples.shape == (B, C, N)
                sample_list.append(samples[:, :3, :].permute(0, 2, 1).detach().cpu().numpy())
                loss /= config.test.ddp.nproc_this_node
                loss_list.append(loss.detach().cpu().numpy())
                pbar.update(i)
    
        if rank == 0:
            if config.test.save_pkl:
                save_statical_data(data_dict, save_dir, statistic_data_all_samples)


if __name__ == '__main__':
    num_arguments = len(sys.argv)

    if num_arguments > 1:
        main_with_Decorators()
    else:
        subprocess.run('nvidia-smi', shell=True, text=True, stdout=None, stderr=subprocess.PIPE)
        config = OmegaConf.load('configs/default.yaml')
        cmd_config = {
            'usr_config': 'configs/seg.yaml',
            'datasets': 'shapenet_AnTao350M',
            'wandb': {'name': '2024_04_16_01_26_Shapenet_Token_Std_logmean_1'},
            'test': {'ddp': {'which_gpu': [3]}}
        }
        config = OmegaConf.merge(config, OmegaConf.create(cmd_config))

        dataset_config = OmegaConf.load(f'configs/datasets/{config.datasets}.yaml')
        dataset_config = OmegaConf.create({'datasets': dataset_config})
        config = OmegaConf.merge(config, dataset_config)

        main_without_Decorators(config)
