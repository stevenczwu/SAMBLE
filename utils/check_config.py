import subprocess
import socket
import random


def check_config(config):
    '''
    Check if there are conflicts in the config
    '''

    idx_mode_dict = {
        "token": ["col_sum", "row_std", "sparse_row_sum", "sparse_row_std", "sparse_col_sum", "sparse_col_avg",
                  "sparse_col_sqr"],
        "global_carve": ["col_sum", "row_std", "sparse_row_sum", "sparse_row_std", "sparse_col_sum", "sparse_col_avg",
                         "sparse_col_sqr"],
        "local": ["local_std"],
        "global": ["col_sum"],
        "local_insert": ["local_std", "sparse_row_std", "sparse_col_sum", "sparse_col_avg", "sparse_col_sqr"],
    }
    cls_datasets = ["modelnet_AnTao420M", "modelnet_Alignment1024"]
    seg_datasets = ["shapenet_AnTao350M", "shapenet_Yi650M", "shapenet_Normal"]

    bin_boundaries = config.feature_learning_block.downsample.bin.bin_boundaries
    num_bins = config.feature_learning_block.downsample.bin.num_bins
    for i, bin_boundaries_one_layer in enumerate(bin_boundaries):
        if config.feature_learning_block.downsample.bin.enable[i] \
                and config.feature_learning_block.downsample.bin.mode[i] == 'nonuniform_split_bin':
            assert num_bins[i] == len(bin_boundaries_one_layer) + 1, 'Length of bin_boundaries should equal num_bins-1.'

    # train & test
    # gpu
    if config.mode == 'train':
        assert config.train.ddp.nproc_this_node == config.train.ddp.world_size == len(
            config.train.ddp.which_gpu), "Train GPU settings must match each other!"
    elif config.mode == 'test':
        assert config.test.ddp.nproc_this_node == config.test.ddp.world_size == len(
            config.test.ddp.which_gpu), "Test GPU settings must match each other!"
    else:
        raise NotImplementedError

    if config.train.epochs <= 50:
        assert config.test.visualize_preds.num_vis <= 5 and config.test.visualize_downsampled_points.num_vis <= 5 and config.test.visualize_attention_heatmap.num_vis <= 5, \
            "When train.epochs is less than 50(debugging), only 5 predictions are allowed!"
    # elif config.train.epochs > 50:
    #     # assert config.test.visualize_preds.num_vis >= 40 and config.test.visualize_downsampled_points.num_vis >= 40 and config.test.visualize_attention_heatmap.num_vis >= 40, \
    #     #     "When train.epochs is 50+, please set at least 40 predictions!"
    #     if config.datasets.dataset_name in cls_datasets:
    #         assert config.train.ddp.nproc_this_node * config.train.dataloader.batch_size_per_gpu >= 8, "Batch size of classification training must bigger than 8!"
    #     elif config.datasets.dataset_name in seg_datasets:
    #         assert config.train.ddp.nproc_this_node * config.train.dataloader.batch_size_per_gpu >= 16, "Batch size of segmentation training must bigger than 16!"
    #     else:
    #         raise ValueError("Unknown dataset!")
    # vote
    if config.train.dataloader.vote.enable:
        assert config.train.dataloader.vote.num_vote >= 2, "When vote is enabled, num_votes should be at least 2!"
        assert config.train.dataloader.vote.vote_start_epoch <= config.train.epochs, "When vote is enabled, vote must start before the end of training!"
        # test
    if config.test.visualize_combine.enable:
        assert config.test.visualize_downsampled_points.num_vis == config.test.visualize_attention_heatmap.num_vis, "If vis_combine is enabled, downsample points and heatmap must be visualized in the same amount!"
        for idx_mode in config.test.visualize_combine.vis_which:
            if idx_mode not in idx_mode_dict[config.feature_learning_block.downsample.ds_which]:
                raise ValueError(
                    f"When visualize_combine is enabled, vis_which should be one of {idx_mode_dict[config.feature_learning_block.downsample.ds_which]}! Got: {idx_mode}")

    # block
    assert config.feature_learning_block.enable and not config.point2point_block.enable and not config.edgeconv_block.enable, "Only N2P block can be enabled!"

    # feature_learning_block.embedding
    if config.feature_learning_block.embedding.normal_channel + (
            config.datasets.dataset_name == "shapenet_Normal") == 1:
        raise ValueError("embedding.normal_channel and dataset shapenet_Normal must be sync setted!")
    elif config.datasets.dataset_name == "shapenet_Normal":
        assert config.feature_learning_block.embedding.conv1_in[
                   0] == 12, "When use dataset with normal vector, the first conv_in must be 12"
    else:
        assert config.feature_learning_block.embedding.conv1_in[
                   0] == 6, "When didn't use dataset with normal vector, the first conv_in must be 6"

        # feature_learning_block.downsample    
    for i in range(len(config.feature_learning_block.downsample.M)):
        q_in = config.feature_learning_block.downsample.q_in[i]
        q_out = config.feature_learning_block.downsample.q_out[i]
        k_in = config.feature_learning_block.downsample.k_in[i]
        k_out = config.feature_learning_block.downsample.k_out[i]
        v_in = config.feature_learning_block.downsample.v_in[i]
        v_out = config.feature_learning_block.downsample.v_out[i]
        num_heads = config.feature_learning_block.downsample.num_heads[i]
        num_bins = config.feature_learning_block.downsample.bin.num_bins[i]
        idx_mode = config.feature_learning_block.downsample.idx_mode[i]
        bin_enable = config.feature_learning_block.downsample.bin.enable[i]
        boltzmann_enable = config.feature_learning_block.downsample.boltzmann.enable[i]

        if bin_enable and boltzmann_enable:
            raise ValueError("bin and boltzmann cannot be enabled at the same time!")

        assert idx_mode in idx_mode_dict[config.feature_learning_block.downsample.ds_which], \
            f"When downsample mode is {config.feature_learning_block.downsample.ds_which}, idx_mode should be one of {idx_mode_dict[config.feature_learning_block.downsample.ds_which]}! Got: {idx_mode}"

        if q_in != k_in or q_in != v_in or k_in != v_in:
            raise ValueError(f'q_in, k_in and v_in should be the same! Got q_in:{q_in}, k_in:{k_in}, v_in:{v_in}')
        if q_out != k_out:
            raise ValueError('q_out should be equal to k_out!')
        if q_out % num_heads != 0 or k_out % num_heads != 0 or v_out % num_heads != 0:
            raise ValueError('please set another value for num_heads!')
        assert num_bins % 2 == 0 and num_bins >= 2, "num_bins should be even and greater than 2!"
        assert num_heads == 1, "num_heads should be 1!"

        # feature_learning_block.attention
    for i in range(len(config.feature_learning_block.attention.K)):
        num_heads = config.feature_learning_block.attention.num_heads[i]
        q_in = config.feature_learning_block.attention.q_in[i]
        q_out = config.feature_learning_block.attention.q_out[i]
        k_in = config.feature_learning_block.attention.k_in[i]
        k_out = config.feature_learning_block.attention.k_out[i]
        v_in = config.feature_learning_block.attention.v_in[i]
        v_out = config.feature_learning_block.attention.v_out[i]
        attention_mode = config.feature_learning_block.attention.attention_mode[i]
        group_type = config.feature_learning_block.attention.group_type[i]
        if q_in != v_out:
            raise ValueError(f'q_in should be equal to v_out due to ResLink! Got q_in: {q_in}, v_out: {v_out}')
        if k_in != v_in:
            raise ValueError(f'k_in and v_in should be the same! Got k_in:{k_in}, v_in:{v_in}')
        if q_out != k_out:
            raise ValueError('q_out should be equal to k_out!')
        if q_out % num_heads != 0 or k_out % num_heads != 0 or v_out % num_heads != 0:
            raise ValueError('please set another value for num_heads!')

        if attention_mode == "scalar_dot":
            assert group_type == "diff", "When attention_mode is scalar_dot, group_type must be diff!"
        elif attention_mode == "vector_sub":
            assert group_type == "neighbor", "When attention_mode is vector_sub, group_type must be neighbor!"

        # feature_learning_block.upsample    
    for i in range(len(config.feature_learning_block.upsample.q_in)):
        q_in = config.feature_learning_block.upsample.q_in[i]
        q_out = config.feature_learning_block.upsample.q_out[i]
        k_in = config.feature_learning_block.upsample.k_in[i]
        k_out = config.feature_learning_block.upsample.k_out[i]
        v_in = config.feature_learning_block.upsample.v_in[i]
        v_out = config.feature_learning_block.upsample.v_out[i]
        num_heads = config.feature_learning_block.upsample.num_heads[i]
        if k_in != v_in:
            raise ValueError(f'k_in and v_in should be the same! Got k_in:{k_in}, v_in:{v_in}')
        if q_out != k_out:
            raise ValueError('q_out should be equal to k_out!')
        if q_out % num_heads != 0 or k_out % num_heads != 0 or v_out % num_heads != 0:
            raise ValueError('please set another value for num_heads!')


def get_gpu_info():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,power.draw,memory.used,utilization.gpu,memory.free',
                                 '--format=csv,noheader,nounits'], stdout=subprocess.PIPE, text=True)
        output = result.stdout.strip().split('\n')
        gpu_info = []
        for line in output:
            gpu_index, power_draw, memory_used, gpu_util, memory_free = line.split(',')
            gpu_info.append({'GPU Index': int(gpu_index), 'Power Draw (W)': float(power_draw),
                             'Memory Used (MB)': float(memory_used), 'GPU Utilization (%)': float(gpu_util),
                             'Memory Free (MB)': float(memory_free)})
        return gpu_info
    except Exception as e:
        return str(e)


def set_available_gpus(gpu_info, num_gpus):
    available_gpus = []
    for gpu in gpu_info:
        if gpu['GPU Utilization (%)'] > 40 or gpu['Memory Free (MB)'] < 11000:
            continue
        else:
            available_gpus.append(gpu['GPU Index'])
    if len(available_gpus) < num_gpus:
        raise ValueError(
            f"Not enough available GPUs! Available GPUs: {available_gpus}, Required number of GPUs: {num_gpus}")
    else:
        available_gpus = available_gpus[:num_gpus]
    return available_gpus


def config_gpus(config, mode):
    gpu_info = get_gpu_info()
    gpus_id = []
    for gpu in gpu_info:
        gpus_id.append(gpu['GPU Index'])

    if mode == "train":
        num_gpus = config.train.ddp.nproc_this_node
        config_gpus = config.train.ddp.which_gpu
    elif mode == "test":
        num_gpus = config.test.ddp.nproc_this_node
        config_gpus = config.test.ddp.which_gpu
    else:
        raise ValueError("Mode should be train or test!")

    # check if config gpus are valid, if not, set other available gpus
    for config_gpu in config_gpus:
        if config_gpu not in gpus_id:
            new_gpus = set_available_gpus(gpu_info, num_gpus)
            if mode == "train":
                config["train"]["ddp"]["which_gpu"] = new_gpus
            else:
                config["test"]["ddp"]["which_gpu"] = new_gpus
            print(f"Invalid GPU ID: {config_gpu}! Now set {mode} config gpus to {new_gpus}!")
            return config

            # check if any gpu is overloaded, if so, set other available gpus
    hostname = socket.gethostname()
    if 'iesservergpu' in hostname:
        for gpu in gpu_info:
            if gpu['GPU Index'] in config_gpus:
                if gpu['GPU Utilization (%)'] > 40 or gpu['Memory Free (MB)'] < 11000:
                    new_gpus = set_available_gpus(gpu_info, num_gpus)
                    if mode == "train":
                        config.train.ddp.which_gpu = new_gpus
                    else:
                        config.test.ddp.which_gpu = new_gpus
                    print(f"GPU {gpu['GPU Index']} is overloaded! Now set {mode} config gpus to {new_gpus}!")
                    break
                else:
                    continue
    else:
        if mode == "train":
            config.train.ddp.which_gpu = [0, 1]
        else:
            config.test.ddp.which_gpu = [0, 1]
        print(f"Set {mode} config gpus to {config.train.ddp.which_gpu}!")

    return config


def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            if port < 0 or port > 65535:
                print(f"Port {port} is out of range!")
                return False
            s.bind(("127.0.0.1", port))
        except OSError:
            print(f"Port {port} is already in use!")
            return False
    return True


def set_port(config, mode):
    if mode == "train":
        port = config.train.ddp.master_port
    elif mode == "test":
        port = config.test.ddp.master_port
    else:
        raise ValueError("Mode should be train or test!")

    if not is_port_available(port):
        num_tmp = 0
        while True:
            new_port = random.randint(10000, 20000)
            if is_port_available(new_port):
                break
            num_tmp += 1
            if num_tmp > 100:
                raise ValueError("Cannot find available port!")
        if mode == "train":
            config.train.ddp.master_port = new_port
        else:
            config.test.ddp.master_port = new_port
        print(f"Now set {mode} config port to {new_port}!")
    return config


def check_worldsize(config, mode):
    if mode == 'train':
        # print(f'which_gpu:{config.train.ddp.which_gpu}')
        # print(f'which_gpu_type{type(config.train.ddp.which_gpu)}')

        import omegaconf
        assert (isinstance(config.train.ddp.which_gpu, omegaconf.listconfig.ListConfig))

        num_gpus = len(config.train.ddp.which_gpu)
        config.train.ddp.nproc_this_node = num_gpus
        config.train.ddp.world_size = num_gpus

    elif mode == 'test':
        import omegaconf
        assert (isinstance(config.test.ddp.which_gpu, omegaconf.listconfig.ListConfig))

        num_gpus = len(config.test.ddp.which_gpu)
        config.test.ddp.nproc_this_node = num_gpus
        config.test.ddp.world_size = num_gpus

    else:
        raise NotImplementedError

    return config


def set_config_run(config, mode, check_config_flag=True):
    # config = config_gpus(config, mode)
    config = check_worldsize(config, mode)
    config = set_port(config, mode)
    if check_config_flag:
        check_config(config)
    return config
