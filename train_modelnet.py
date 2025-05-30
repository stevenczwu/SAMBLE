import shutil
from utils import dataloader, lr_scheduler
from models import cls_model
from omegaconf import OmegaConf
import hydra
from pathlib import Path
import torch
import pkbar
import wandb
from utils import metrics
import os
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.cuda import amp
import numpy as np
from utils.loss import consistency_loss, aux_loss
from utils.check_config import set_config_run
import time
import datetime
import socket
import sys
import subprocess


@hydra.main(version_base=None, config_path="./configs", config_name="default.yaml")
def main(config):
    main_without_Decorators(config)


def main_without_Decorators(config):
    # check working directory
    try:
        assert str(Path.cwd().resolve()) == str(Path(__file__).resolve().parents[0])
    except:
        exit(f'Working directory is not the same as project root. Exit.')

    # overwrite the default config with user config
    config.mode = 'train'
    if config.usr_config:
        usr_config = OmegaConf.load(config.usr_config)
        config = OmegaConf.merge(config, usr_config)
    config = set_config_run(config, "train")

    if config.datasets.dataset_name == 'modelnet':
        dataloader.download_modelnet(config.datasets.url, config.datasets.saved_path)
    else:
        raise ValueError('Not implemented!')

    # multiprocessing for ddp
    if config.train.ddp.random_seed == 0:
        random_seed = int(time.time())
    else:
        random_seed = config.train.ddp.random_seed

    time_label = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')

    print(f'cuda available: {torch.cuda.is_available()}')

    if torch.cuda.is_available():
        os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'  # read .h5 file using multiprocessing will raise error
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.train.ddp.which_gpu).replace(' ', '').replace('[', '').replace(
            ']', '')
        mp.spawn(train, args=(config, random_seed, time_label), nprocs=config.train.ddp.nproc_this_node, join=True)
    else:
        exit('It is almost impossible to train this model using CPU. Please use GPU! Exit.')


def train(local_rank, config, random_seed,
          time_label):  # the first arg must be local rank for the sake of using mp.spawn(...)

    rank = config.train.ddp.rank_starts_from + local_rank

    torch.manual_seed(random_seed)

    if config.wandb.enable and rank == 0:
        hostname = socket.gethostname()
        config.wandb.name = f'{time_label}_{config.wandb.name}'
        if 'iesservergpu' in hostname:
            save_dir = '/home/ies/fu/train_output/'  # '/data/users/fu/APES'
        else:
            save_dir = '/home/team1/cwu/FuHaoWorkspace/APES/'

        # initialize wandb
        wandb.login(key=config.wandb.api_key)
        del config.wandb.api_key, config.test
        config_dict = OmegaConf.to_container(config, resolve=True)
        run = wandb.init(project=config.wandb.project, entity=config.wandb.entity, config=config_dict,
                         name=config.wandb.name)
        # cache source code for saving
        os.makedirs(f'{save_dir}{time_label}_{run.id}/models')
        os.makedirs(f'{save_dir}{time_label}_{run.id}/utils')

        OmegaConf.save(config=config, f=f'{save_dir}{time_label}_{run.id}/usr_config.yaml', resolve=False)
        os.system(f'cp ./models/cls_model.py {save_dir}{time_label}_{run.id}/models/cls_model.py')
        os.system(f'cp ./models/cls_block.py {save_dir}{time_label}_{run.id}/models/cls_block.py')
        os.system(f'cp ./models/attention.py {save_dir}{time_label}_{run.id}/models/attention.py')
        os.system(f'cp ./models/downsample.py {save_dir}{time_label}_{run.id}/models/downsample.py')
        os.system(f'cp ./models/embedding.py {save_dir}{time_label}_{run.id}/models/embedding.py')
        os.system(f'cp ./utils/dataloader.py {save_dir}{time_label}_{run.id}/utils/dataloader.py')
        os.system(f'cp ./utils/metrics.py {save_dir}{time_label}_{run.id}/utils/metrics.py')
        os.system(f'cp ./utils/ops.py {save_dir}{time_label}_{run.id}/utils/ops.py')
        os.system(
            f'cp ./utils/data_augmentation.py {save_dir}{time_label}_{run.id}/utils/data_augmentation.py')
        os.system(f'cp ./utils/debug.py {save_dir}{time_label}_{run.id}/utils/debug.py')
        os.system(f'cp ./utils/check_config.py {save_dir}{time_label}_{run.id}/utils/check_config.py')
        os.system(f'cp ./utils/save_backup.py {save_dir}{time_label}_{run.id}/utils/save_backup.py')
        os.system(f'cp ./utils/visualization.py {save_dir}{time_label}_{run.id}/utils/visualization.py')
        os.system(
            f'cp ./utils/visualization_data_processing.py {save_dir}{time_label}_{run.id}/utils/visualization_data_processing.py')
        os.system(f'cp ./utils/lr_scheduler.py {save_dir}{time_label}_{run.id}/utils/lr_scheduler.py')
        os.system(f'cp ./train_modelnet.py {save_dir}{time_label}_{run.id}/train_modelnet.py')
        os.system(f'cp ./test_modelnet.py {save_dir}{time_label}_{run.id}/test_modelnet.py')
        with open(f'{save_dir}{time_label}_{run.id}/random_seed_{random_seed}.txt', 'w') as f:
            f.write('')

    # process initialization
    os.environ['MASTER_ADDR'] = str(config.train.ddp.master_addr)
    os.environ['MASTER_PORT'] = str(config.train.ddp.master_port)
    os.environ['WORLD_SIZE'] = str(config.train.ddp.world_size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    # gpu setting
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)  # which gpu is used by current process
    print(
        f'[init] pid: {os.getpid()} - global rank: {rank} - local rank: {local_rank} - cuda: {config.train.ddp.which_gpu[local_rank]}')

    # create a scaler for amp
    scaler = amp.GradScaler()

    # get dataset
    if config.datasets.dataset_name == 'modelnet':
        trainval_set, test_set = dataloader.get_modelnet_dataset(config.datasets.saved_path,
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
                                                                           config.train.dataloader.vote.enable,
                                                                           config.train.dataloader.vote.num_vote)
    else:
        raise ValueError('Not implemented!')

    # get sampler
    trainval_sampler = torch.utils.data.distributed.DistributedSampler(trainval_set)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)

    # get dataloader
    trainval_loader = torch.utils.data.DataLoader(trainval_set, config.train.dataloader.batch_size_per_gpu,
                                                  num_workers=config.train.dataloader.num_workers, drop_last=True,
                                                  prefetch_factor=config.train.dataloader.prefetch,
                                                  pin_memory=config.train.dataloader.pin_memory,
                                                  sampler=trainval_sampler)
    test_loader = torch.utils.data.DataLoader(test_set, config.train.dataloader.batch_size_per_gpu,
                                              num_workers=config.train.dataloader.num_workers, drop_last=True,
                                              prefetch_factor=config.train.dataloader.prefetch,
                                              pin_memory=config.train.dataloader.pin_memory, sampler=test_sampler)

    # if combine train and validation
    if config.train.dataloader.combine_trainval:
        train_sampler = trainval_sampler
        train_loader = trainval_loader
        validation_loader = test_loader
    else:
        raise ValueError(
            'modelnet40 has only train_set and test_set, which means validation_set is included in train_set!')

    # get model
    my_model = cls_model.ModelNetModel(config)

    # synchronize bn among gpus
    if config.train.ddp.syn_bn:  
        my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)

    # get ddp model
    my_model = my_model.to(device)
    my_model = torch.nn.parallel.DistributedDataParallel(my_model)

    # get optimizer
    if config.train.optimizer.which == 'adamw':
        optimizer = torch.optim.AdamW(my_model.parameters(), lr=config.train.lr,
                                      weight_decay=config.train.optimizer.weight_decay)
    elif config.train.optimizer.which == 'sgd':
        optimizer = torch.optim.SGD(my_model.parameters(), lr=config.train.lr,
                                    weight_decay=config.train.optimizer.weight_decay, momentum=0.9)
    else:
        raise ValueError('Not implemented!')

    # get lr scheduler
    if config.train.lr_scheduler.enable:
        if config.train.lr_scheduler.which == 'stepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                        step_size=config.train.lr_scheduler.stepLR.decay_step,
                                                        gamma=config.train.lr_scheduler.stepLR.gamma)
        elif config.train.lr_scheduler.which == 'expLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.train.lr_scheduler.expLR.gamma)
        elif config.train.lr_scheduler.which == 'cosLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=config.train.lr_scheduler.cosLR.T_max,
                                                                   eta_min=config.train.lr_scheduler.cosLR.eta_min)
        elif config.train.lr_scheduler.which == 'cos_warmupLR':
            scheduler = lr_scheduler.CosineAnnealingWithWarmupLR(optimizer,
                                                                 T_max=config.train.lr_scheduler.cos_warmupLR.T_max,
                                                                 eta_min=config.train.lr_scheduler.cos_warmupLR.eta_min,
                                                                 warmup_init_lr=config.train.lr_scheduler.cos_warmupLR.warmup_init_lr,
                                                                 warmup_epochs=config.train.lr_scheduler.cos_warmupLR.warmup_epochs)
        else:
            raise ValueError('Not implemented!')

    # get loss function
    if config.train.label_smoothing:
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean', label_smoothing=config.train.epsilon)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    val_acc_list = [0]
    # start training
    for epoch in range(config.train.epochs):
        my_model.train()
        train_sampler.set_epoch(epoch)
        train_loss_list = []
        pred_list = []
        cls_label_list = []
        if rank == 0:
            kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=config.train.epochs,
                              always_stateful=True)
        for i, (samples, cls_labels) in enumerate(train_loader):
            optimizer.zero_grad()
            samples, cls_labels = samples.to(device), cls_labels.to(device)
            if config.train.amp:
                with amp.autocast():
                    preds = my_model(samples)
                    if config.train.aux_loss.enable:
                        train_loss = 0
                        for pred in preds:
                            train_loss += loss_fn(pred, cls_labels)
                        train_loss = train_loss / len(preds)
                        preds = preds[-1]
                    else:
                        train_loss = loss_fn(preds,
                                             cls_labels) + config.train.consistency_loss_factor * consistency_loss(
                            my_model.module.block.res_link_list)
                scaler.scale(train_loss).backward()
                
                if config.train.grad_clip.enable:
                    scaler.unscale_(optimizer)
                    if config.train.grad_clip.mode == 'value':
                        torch.nn.utils.clip_grad_value_(my_model.parameters(), config.train.grad_clip.value)
                    elif config.train.grad_clip.mode == 'norm':
                        torch.nn.utils.clip_grad_norm_(my_model.parameters(), config.train.grad_clip.max_norm)
                    else:
                        raise ValueError('mode should be value or norm!')
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = my_model(samples)

                if config.feature_learning_block.res_link.enable:
                    if config.train.aux_loss.enable:
                        train_loss = loss_fn(preds[-1], cls_labels) + config.train.aux_loss.factor * aux_loss(preds,
                                                                                                              cls_labels,
                                                                                                              loss_fn) + config.train.consistency_loss_factor * consistency_loss(
                            my_model.module.block.res_link_list)
                        preds = preds[-1]
                    else:
                        if config.train.consistency_loss_factor > 0:
                            train_loss = (
                                    loss_fn(preds, cls_labels) +
                                    config.train.consistency_loss_factor * consistency_loss(
                                my_model.module.block.res_link_list))
                        else:
                            train_loss = loss_fn(preds, cls_labels)
                else:
                    assert config.train.aux_loss.enable == False and config.train.consistency_loss_factor == 0, "If there is no residual link in the structure, consistency loss and auxiliary loss must be False!"
                    train_loss = loss_fn(preds, cls_labels)
                train_loss.backward()
                
                if config.train.grad_clip.enable:
                    if config.train.grad_clip.mode == 'value':
                        torch.nn.utils.clip_grad_value_(my_model.parameters(), config.train.grad_clip.value)
                    elif config.train.grad_clip.mode == 'norm':
                        torch.nn.utils.clip_grad_norm_(my_model.parameters(), config.train.grad_clip.max_norm)
                    else:
                        raise ValueError('mode should be value or norm!')
                optimizer.step()

            # collect the result from all gpus
            pred_gather_list = [torch.empty_like(preds).to(device) for _ in range(config.train.ddp.nproc_this_node)]
            cls_label_gather_list = [torch.empty_like(cls_labels).to(device) for _ in
                                     range(config.train.ddp.nproc_this_node)]
            torch.distributed.all_gather(pred_gather_list, preds)
            torch.distributed.all_gather(cls_label_gather_list, cls_labels)
            torch.distributed.all_reduce(train_loss)
            if rank == 0:
                preds = torch.concat(pred_gather_list, dim=0)
                pred_list.append(torch.max(preds, dim=1)[1].detach().cpu().numpy())
                cls_labels = torch.concat(cls_label_gather_list, dim=0)
                cls_label_list.append(torch.max(cls_labels, dim=1)[1].detach().cpu().numpy())
                train_loss /= config.train.ddp.nproc_this_node
                train_loss_list.append(train_loss.detach().cpu().numpy())
                kbar.update(i)

        # decay lr
        current_lr = optimizer.param_groups[0]['lr']
        if config.train.lr_scheduler.enable:
            if config.train.lr_scheduler.which == 'cosLR' and epoch + 1 > config.train.lr_scheduler.cosLR.T_max:
                pass
            else:
                scheduler.step()

        # calculate metrics
        if rank == 0:
            preds = np.concatenate(pred_list, axis=0)
            cls_labels = np.concatenate(cls_label_list, axis=0)
            train_acc = metrics.calculate_accuracy(preds, cls_labels)
            train_loss = sum(train_loss_list) / len(train_loss_list)

        # log results
        if rank == 0:
            if config.wandb.enable:
                metric_dict = {'modelnet_train': {'lr': current_lr, 'loss': train_loss, 'acc': train_acc}}
                if (epoch + 1) % config.train.validation_freq:
                    wandb.log(metric_dict, commit=True)
                else:
                    wandb.log(metric_dict, commit=False)

        # start validation
        if not (epoch + 1) % config.train.validation_freq:
            my_model.eval()
            val_loss_list = []
            pred_list = []
            cls_label_list = []
            with torch.no_grad():
                for samples, cls_labels in validation_loader:
                    cls_labels = cls_labels.to(device)
                    if config.train.dataloader.vote.enable:
                        if epoch + 1 >= config.train.dataloader.vote.vote_start_epoch:
                            preds_list = []
                            for samples_vote in samples:
                                samples_vote = samples_vote.to(device)
                                preds = my_model(samples_vote)
                                preds_list.append(preds)
                            preds = torch.mean(torch.stack(preds_list), dim=0)
                        else:
                            samples = samples[0].to(device)
                            preds = my_model(samples)
                    else:
                        samples = samples.to(device)
                        preds = my_model(samples)

                    if config.train.aux_loss.enable:
                        val_loss = loss_fn(preds[-1], cls_labels)
                        preds = preds[-1]
                    else:
                        val_loss = loss_fn(preds, cls_labels)

                    # collect the result among all gpus
                    pred_gather_list = [torch.empty_like(preds).to(device) for _ in
                                        range(config.train.ddp.nproc_this_node)]
                    cls_label_gather_list = [torch.empty_like(cls_labels).to(device) for _ in
                                             range(config.train.ddp.nproc_this_node)]
                    torch.distributed.all_gather(pred_gather_list, preds)
                    torch.distributed.all_gather(cls_label_gather_list, cls_labels)
                    torch.distributed.all_reduce(val_loss)
                    if rank == 0:
                        preds = torch.concat(pred_gather_list, dim=0)
                        pred_list.append(torch.max(preds, dim=1)[1].detach().cpu().numpy())
                        cls_labels = torch.concat(cls_label_gather_list, dim=0)
                        cls_label_list.append(torch.max(cls_labels, dim=1)[1].detach().cpu().numpy())
                        val_loss /= config.train.ddp.nproc_this_node
                        val_loss_list.append(val_loss.detach().cpu().numpy())

            # calculate metrics
            if rank == 0:
                preds = np.concatenate(pred_list, axis=0)
                cls_labels = np.concatenate(cls_label_list, axis=0)
                val_acc = metrics.calculate_accuracy(preds, cls_labels)
                val_loss = sum(val_loss_list) / len(val_loss_list)

            # log results
            if rank == 0:
                kbar.update(i + 1, values=[('lr', current_lr), ('train_loss', train_loss), ('train_acc', train_acc),
                                           ('val_loss', val_loss), ('val_acc', val_acc)])
                if config.wandb.enable:
                    # save model
                    if val_acc >= max(val_acc_list):

                        if config.feature_learning_block.downsample.bin.dynamic_boundaries:

                            state_dict = {'model_state_dict': my_model.state_dict(),
                                          'bin_boundaries': [downsample_module.bin_boundaries for downsample_module in
                                                             my_model.module.block.downsample_list]}
                        else:
                            state_dict = my_model.state_dict()

                        torch.save(state_dict, f'{save_dir}{time_label}_{run.id}/checkpoint.pt')
                    val_acc_list.append(val_acc)
                    metric_dict = {'modelnet_val': {'loss': val_loss, 'acc': val_acc}}
                    metric_dict['modelnet_val']['best_acc'] = max(val_acc_list)
                    wandb.log(metric_dict, commit=True)
        else:
            if rank == 0:
                kbar.update(i + 1, values=[('lr', current_lr), ('train_loss', train_loss), ('train_acc', train_acc)])

    # save artifacts to wandb server
    if config.wandb.enable and rank == 0:
        artifacts = wandb.Artifact(config.wandb.name, type='runs')
        artifacts.add_file(f'{save_dir}{time_label}_{run.id}/usr_config.yaml', name='usr_config.yaml')
        artifacts.add_dir(f"{save_dir}{time_label}_{run.id}/models", name='models')
        artifacts.add_dir(f"{save_dir}{time_label}_{run.id}/utils", name='utils')
        artifacts.add_file(f'{save_dir}{time_label}_{run.id}/train_modelnet.py', name='train_modelnet.py')
        artifacts.add_file(f'{save_dir}{time_label}_{run.id}/test_modelnet.py', name='test_modelnet.py')
        artifacts.add_file(f'{save_dir}{time_label}_{run.id}/checkpoint.pt', name='checkpoint.pt')
        run.log_artifact(artifacts)
        wandb.finish(quiet=True)
        artifact_name = artifacts.digest
        print("Artifact name:", artifact_name)


if __name__ == '__main__':
    num_arguments = len(sys.argv)

    if num_arguments > 1:
        main()
    else:
        subprocess.run('nvidia-smi', shell=True, text=True, stdout=None, stderr=subprocess.PIPE)
        config = OmegaConf.load('configs/default.yaml')
        cmd_config = {
            'train': {'epochs': 200, 'ddp': {'which_gpu': [3]}},
            'datasets': 'modelnet', 
            'usr_config': 'configs/cls.yaml',
            'wandb': {'name': 'Test'}
        }
        config = OmegaConf.merge(config, OmegaConf.create(cmd_config))

        dataset_config = OmegaConf.load(f'configs/datasets/{config.datasets}.yaml')
        dataset_config = OmegaConf.create({'datasets': dataset_config})
        config = OmegaConf.merge(config, dataset_config)
        # print(config.datasets)

        main_without_Decorators(config)
