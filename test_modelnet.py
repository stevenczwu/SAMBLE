import wandb
from omegaconf import OmegaConf
from pathlib import Path
from utils import dataloader
from models import cls_model
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import sys
import hydra
import subprocess
import pkbar
import os

from utils.ops import reshape_gathered_variable, gather_variable_from_gpus
from utils.check_config import set_config_run


@hydra.main(version_base=None, config_path="./configs", config_name="default.yaml")
def main(config):
    main_without_Decorators(config)


def main_without_Decorators(config):
    # check working directory
    try:
        assert str(Path.cwd().resolve()) == str(Path(__file__).resolve().parents[0])
    except:
        exit(f"Working directory is not the same as project root. Exit.")

    # get test configurations
    if config.usr_config:
        test_config = OmegaConf.load(config.usr_config)
        config = OmegaConf.merge(config, test_config)

    # download artifacts
    if config.wandb.enable:
        wandb.login(key=config.wandb.api_key)
        api = wandb.Api()
        artifact = api.artifact(
            f"{config.wandb.entity}/{config.wandb.project}/{config.wandb.name}:latest"
        )
        if config.test.suffix.enable:
            local_path = f"./artifacts/{config.wandb.name}_{config.test.suffix.remark}"
        else:
            local_path = f"./artifacts/{config.wandb.name}"
        artifact.download(root=local_path)
    else:
        raise ValueError("W&B is not enabled!")

    # overwrite the default config with previous run config
    config.mode = "test"
    run_config = OmegaConf.load(f"{local_path}/usr_config.yaml")
    if not config.test.suffix.enable:
        config = OmegaConf.merge(config, run_config)
    else:
        OmegaConf.save(config, f"{local_path}/usr_config_test.yaml")
        print(f"Overwrite the previous run config with new run config.")
    config = set_config_run(config, "test")

    if config.datasets.dataset_name == "modelnet":
        dataloader.download_modelnet(config.datasets.url, config.datasets.saved_path)
    else:
        raise ValueError("Not implemented!")

    # multiprocessing for ddp
    if torch.cuda.is_available():
        os.environ[
            "HDF5_USE_FILE_LOCKING"
        ] = "FALSE"  # read .h5 file using multiprocessing will raise error
        os.environ["CUDA_VISIBLE_DEVICES"] = (
            str(config.test.ddp.which_gpu)
            .replace(" ", "")
            .replace("[", "")
            .replace("]", "")
        )
        mp.spawn(
            test, args=(config,), nprocs=config.test.ddp.nproc_this_node, join=True
        )
    else:
        raise ValueError("Please use GPU for testing!")


def test(local_rank, config):
    rank = config.test.ddp.rank_starts_from + local_rank

    save_dir = f"/Your/Save/Dir/{config.wandb.name}/"
    if rank == 0:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    # set files path
    if config.test.suffix.enable:
        artifacts_path = f"./artifacts/{config.wandb.name}_{config.test.suffix.remark}"
    else:
        artifacts_path = f"./artifacts/{config.wandb.name}"

    # process initialization
    os.environ["MASTER_ADDR"] = str(config.test.ddp.master_addr)
    os.environ["MASTER_PORT"] = str(config.test.ddp.master_port)
    os.environ["WORLD_SIZE"] = str(config.test.ddp.world_size)
    os.environ["RANK"] = str(rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    # gpu setting
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)  # which gpu is used by current process
    print(
        f"[init] pid: {os.getpid()} - global rank: {rank} - local rank: {local_rank} - cuda: {config.test.ddp.which_gpu[local_rank]}"
    )

    # get datasets
    if config.datasets.dataset_name == "modelnet":
        # _, test_set = dataloader.get_modelnet_dataset(config.datasets.saved_path,
        _, test_set = dataloader.get_modelnet_dataset(
            config.datasets.saved_path,
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
        )
    else:
        raise ValueError("Not implemented!")

    # get sampler
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)

    # get dataloader
    test_loader = torch.utils.data.DataLoader(
        test_set,
        config.test.dataloader.batch_size_per_gpu,
        num_workers=config.test.dataloader.num_workers,
        drop_last=True,
        prefetch_factor=config.test.dataloader.prefetch,
        pin_memory=config.test.dataloader.pin_memory,
        sampler=test_sampler,
    )

    # get model
    my_model = cls_model.ModelNetModel(config)
    my_model.eval()
    my_model = my_model.to(device)
    my_model = torch.nn.parallel.DistributedDataParallel(my_model)
    map_location = {"cuda:0": f"cuda:{local_rank}"}

    if config.feature_learning_block.downsample.bin.dynamic_boundaries:
        state_dict = torch.load(
            f"{artifacts_path}/checkpoint.pt", map_location=map_location
        )
        my_model.load_state_dict(state_dict["model_state_dict"])

        config.feature_learning_block.downsample.bin.dynamic_boundaries = False
        config.feature_learning_block.downsample.bin.bin_boundaries = [
            bin_boundaries[0][0, 0, 0, 1:].tolist()
            for bin_boundaries in state_dict["bin_boundaries"]
        ]
    else:
        my_model.load_state_dict(
            torch.load(f"{artifacts_path}/checkpoint.pt", map_location=map_location)
        )

    # get loss function
    if config.test.label_smoothing:
        loss_fn = torch.nn.CrossEntropyLoss(
            reduction="mean", label_smoothing=config.test.epsilon
        )
    else:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

    # start test
    loss_list = []
    pred_list = []
    cls_label_list = []
    sample_list = []

    with torch.no_grad():
        if rank == 0:
            print(
                f"Print Results: {config.test.print_results} - Visualize Downsampled Points: {config.test.visualize_downsampled_points.enable} - Visualize Heatmap: {config.test.visualize_attention_heatmap.enable}"
            )
            pbar = pkbar.Pbar(
                name="Start testing, please wait...", target=len(test_loader)
            )

        for i, (samples, cls_labels) in enumerate(test_loader):
            samples, cls_labels = samples.to(device), cls_labels.to(device)
            preds = my_model(samples)

            loss = loss_fn(preds, cls_labels)

            # collect the result among all gpus
            pred_gather_list = [
                torch.empty_like(preds).to(device)
                for _ in range(config.test.ddp.nproc_this_node)
            ]
            cls_label_gather_list = [
                torch.empty_like(cls_labels).to(device)
                for _ in range(config.test.ddp.nproc_this_node)
            ]

            samples = samples.permute(0, 2, 1).contiguous()  # samples: (B,3,N)->(B,N,3)
            sample_gather_list = [
                torch.empty_like(samples).to(device)
                for _ in range(config.test.ddp.nproc_this_node)
            ]

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

                for i_layer, downsample_module in enumerate(
                    my_model.module.block.downsample_list
                ):
                    downsample_module.output_variable_calculatio()

                    sampling_score_all_layers.append(
                        gather_variable_from_gpus(
                            downsample_module,
                            "attention_point_score",
                            rank,
                            config.test.ddp.nproc_this_node,
                            device,
                        )
                    )

                    idx_down_all_layers.append(
                        gather_variable_from_gpus(
                            downsample_module,
                            "idx",
                            rank,
                            config.test.ddp.nproc_this_node,
                            device,
                        )
                    )

                    idx_in_bins_all_layers.append(
                        gather_variable_from_gpus(
                            downsample_module,
                            "idx_chunks",
                            rank,
                            config.test.ddp.nproc_this_node,
                            device,
                        )
                    )
                    k_point_to_choose_all_layers.append(
                        gather_variable_from_gpus(
                            downsample_module,
                            "k_point_to_choose",
                            rank,
                            config.test.ddp.nproc_this_node,
                            device,
                        )
                    )

                    bin_prob = gather_variable_from_gpus(
                        downsample_module,
                        "bin_prob",
                        rank,
                        config.test.ddp.nproc_this_node,
                        device,
                    )
                    # bin_prob.shape == (B, num_bins)

                if rank == 0:
                    # sampling_score_all_layers: num_layers * (B,H,N) -> (B, num_layers, H, N)
                    sampling_score = reshape_gathered_variable(
                        sampling_score_all_layers
                    )
                    # idx_down_all_layers: num_layers * (B,H,M) -> (B, num_layers, H, N)
                    idx_down = reshape_gathered_variable(idx_down_all_layers)
                    # idx_in_bins_all_layers: num_layers * (B,num_bins,1,n) or num_layers * B * num_bins * (1,n) -> (B, num_layers, num_bins, H, n) or B * num_layers * num_bins * (H,n)
                    idx_in_bins = reshape_gathered_variable(idx_in_bins_all_layers)
                    # probability_of_bins_all_layers: num_layers * (B, num_bins) -> (B, num_layers, num_bins)
                    k_point_to_choose = reshape_gathered_variable(
                        k_point_to_choose_all_layers
                    )

                    num_batches = len(k_point_to_choose)
                    num_layers = len(k_point_to_choose[0])
                    num_bins = len(k_point_to_choose[0][0])
                    probability_of_bins = torch.empty(
                        (num_batches, num_layers, num_bins), dtype=torch.float
                    )
                    for i0 in range(num_batches):
                        for j0 in range(num_layers):
                            for k0 in range(num_bins):
                                probability_of_bins[i0, j0, k0] = (
                                    k_point_to_choose[i0][j0][k0]
                                    / idx_in_bins[i0][j0][k0].nelement()
                                )

                    data_dict = {
                        "sampling_score": sampling_score,  # (B, num_layers, H, N)
                        "samples": torch.concat(sample_gather_list, dim=0),  # (B,N,3)
                        "idx_down": idx_down,  # B * num_layers * (H,N)
                        "idx_in_bins": idx_in_bins,
                        # (B, num_layers, num_bins, H, n) or B * num_layers * num_bins * (H,n)
                        "probability_of_bins": probability_of_bins,
                        # B * num_layers * (num_bins)
                        "ground_truth": torch.argmax(
                            torch.concat(cls_label_gather_list, dim=0), dim=1
                        ),
                        # (B,)
                        "predictions": torch.argmax(
                            torch.concat(pred_gather_list, dim=0), dim=1
                        ),  # (B,)
                        "config": config,
                        "raw_learned_bin_prob": bin_prob,
                    }

            if rank == 0:
                preds = torch.concat(pred_gather_list, dim=0)
                pred_list.append(torch.max(preds, dim=1)[1].detach().cpu().numpy())
                cls_labels = torch.concat(cls_label_gather_list, dim=0)
                cls_label_list.append(
                    torch.max(cls_labels, dim=1)[1].detach().cpu().numpy()
                )
                samples = torch.concat(sample_gather_list, dim=0)
                sample_list.append(samples.permute(0, 2, 1).detach().cpu().numpy())
                loss /= config.test.ddp.nproc_this_node
                loss_list.append(loss.detach().cpu().numpy())
                pbar.update(i)


if __name__ == "__main__":
    num_arguments = len(sys.argv)

    if num_arguments > 1:
        main()
    else:
        subprocess.run(
            "nvidia-smi", shell=True, text=True, stdout=None, stderr=subprocess.PIPE
        )
        config = OmegaConf.load("configs/default.yaml")
        cmd_config = {
            "usr_config": "configs/cls.yaml",
            "datasets": "modelnetM",
            "wandb": {
                "name": "2024_04_09_13_39_Modelnet_Token_Std_boltzmann_T0102_norm_sparsesum1_1"
            },
            "test": {"ddp": {"which_gpu": [3]}},
        }
        config = OmegaConf.merge(config, OmegaConf.create(cmd_config))

        dataset_config = OmegaConf.load(f"configs/datasets/{config.datasets}.yaml")
        dataset_config = OmegaConf.create({"datasets": dataset_config})
        config = OmegaConf.merge(config, dataset_config)

        main_without_Decorators(config)
