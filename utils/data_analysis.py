import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import scipy
import torch
import matplotlib.pyplot as plt


def maxwell_pdf(v, sigma):
    return np.sqrt(2 / np.pi) * (v ** 2 * np.exp(-v ** 2 / (2 * sigma ** 2))) / sigma ** 3


def negative_log_likelihood(params, data):
    sigma = params[0]
    likelihoods = maxwell_pdf(data, sigma)
    print(-np.mean(np.log(likelihoods)))
    return -np.mean(np.log(likelihoods))


def _estimate_sigma():
    layer_to_visualize = 0

    tensor = torch.load('modelnet_sampling_scores.pt')
    num_batch = tensor.shape[0]
    tensor = tensor[:, layer_to_visualize + 3, :]

    # tensor = torch.reshape(tensor, (-1,))
    # tensor = tensor[tensor > -1.5]
    # print(tensor[tensor == 0])
    # exit(-1)

    if layer_to_visualize != 0:
        tensor = torch.reshape(tensor, (-1,))
        tensor = tensor[tensor > -1.5]
        tensor = torch.reshape(tensor, (num_batch, 1024))

    # tensor = (tensor - torch.mean(tensor, dim=1, keepdim=True)) / torch.std(tensor, dim=1, unbiased=False, keepdim=True)
    tensor = tensor / torch.std(tensor, dim=1, unbiased=False, keepdim=True)

    data = tensor.flatten().cpu().numpy()

    initial_guess = [1]
    result = minimize(negative_log_likelihood, initial_guess, args=(data,))
    estimated_sigma = result.x[0]

    print(f"estimated sigma: {estimated_sigma}")


def estimate_sigma(tensor=torch.load('modelnet_sampling_scores.pt'), layer_to_visualize=0):
    num_batch = tensor.shape[0]
    tensor = tensor[:, layer_to_visualize + 3, :]

    if layer_to_visualize != 0:
        tensor = torch.reshape(tensor, (-1,))
        tensor = tensor[tensor > -1.5]
        tensor = torch.reshape(tensor, (num_batch, 1024))

    tensor = tensor / torch.std(tensor, dim=1, unbiased=False, keepdim=True)

    data = tensor.flatten().cpu().numpy()

    sigma = np.sqrt(np.pi / 8) * np.mean(data)

    print(f'estimated sigma:{sigma}')

    return sigma


def chi_square_test(tensor=torch.load('modelnet_sampling_scores.pt'), normal_ditribution_test=False):
    if normal_ditribution_test:
        data = torch.normal(0, 1, size=(100000,))

        data = torch.nn.functional.softmax(data, dim=0).cpu().numpy()

        estimated_sigma = np.sqrt(np.pi / 8) * np.mean(data)

        bins = np.linspace(0, estimated_sigma * 10, 11)
    else:

        layer_to_visualize = 0

        estimated_sigma = estimate_sigma(tensor, layer_to_visualize=layer_to_visualize)

        num_batch = tensor.shape[0]
        tensor = tensor[:, layer_to_visualize + 3, :]

        if layer_to_visualize != 0:
            tensor = torch.reshape(tensor, (-1,))
            tensor = tensor[tensor > -1.5]
            tensor = torch.reshape(tensor, (num_batch, 1024))

        tensor = tensor / torch.std(tensor, dim=1, unbiased=False, keepdim=True)
        data = tensor.flatten().cpu().numpy()

        bins = np.linspace(0, 8, 201)
    observed_counts, _ = np.histogram(data, bins)
    observed_counts = observed_counts / np.sum(observed_counts)  # / len(data)

    cdf_values = scipy.stats.maxwell.cdf(bins, scale=estimated_sigma)
    expected_counts = np.diff(cdf_values)  # * len(data)

    # cdf_values = scipy.stats.maxwell.cdf(bins, scale=estimated_sigma * 2)
    # observed_counts = np.diff(cdf_values)

    # print(observed_counts)
    # print(expected_counts)

    bar_width = 0.35
    index = np.arange(len(observed_counts))
    plt.bar(index, observed_counts, bar_width, color='blue', label='observed_distribution')
    plt.bar(index + bar_width, expected_counts, bar_width, color='red', label='expected_Maxwell-Boltzmann_distribution')
    plt.legend()
    plt.show()

    chi_square, p_value = scipy.stats.chisquare(observed_counts, expected_counts)  # expected_counts)

    print(f'p_value:{p_value}')


def visualization_sampling_score(saved_sampling_score_dir='modelnet_sampling_scores.pt', layer_to_visualize=0,
                                 z_normalization_miu=True, show_plt=False,
                                 save_dir=r'C:/Users/Lenovo/Desktop/SamplingScore/', idx=None):
    tensor = torch.load(saved_sampling_score_dir)
    num_batch = tensor.shape[0]
    print(tensor.shape)

    if idx is None:
        tensor = tensor[:, layer_to_visualize + 3, :]
    else:
        tensor = tensor[idx, layer_to_visualize + 3, :].reshape(1, -1)

    # Example tensor of shape [2464, 5, 2048]

    # Flatten the tensor to 1D for histogram

    if layer_to_visualize != 0:
        tensor = torch.reshape(tensor, (-1,))
        tensor = tensor[tensor > -1.5]
        tensor = torch.reshape(tensor, (num_batch, 1024))

    # tensor = (tensor - torch.min(tensor, dim=1, keepdim=True)[0]) / (torch.max(tensor, dim=1, keepdim=True)[0] - torch.min(tensor, dim=1, keepdim=True)[0] + 1e-8)
    if z_normalization_miu:
        tensor = (tensor - torch.mean(tensor, dim=1, keepdim=True)) \
            # / torch.std(tensor, dim=1, unbiased=False, keepdim=True)
        # tensor = tensor / torch.std(tensor, dim=1, unbiased=False, keepdim=True)
    else:
        tensor = (tensor - torch.mean(tensor, dim=1,
                                      keepdim=True))

    flattened_tensor = tensor.flatten().cpu()
    # flattened_tensor = flattened_tensor[flattened_tensor > -1.5]

    min_value = float(torch.min(flattened_tensor))

    topk_values, _ = torch.topk(flattened_tensor, int(flattened_tensor.shape[0] * 0.003), largest=True)
    max_value_9772 = topk_values[-1].item()

    # max_value = float(np.max(flattened_tensor))

    # hist=torch.histc(flattened_tensor,bins=1000,min=min_value,max=max_value)
    # print(hist.)

    # Plotting the histogram
    print(f'min:{min_value}')
    print(f'max:{max_value_9772}')
    plt.figure()
    plt.hist(flattened_tensor, bins=200, range=(min_value, max_value_9772))  # (-2, 4))
    plt.title("Histogram of Tensor Elements")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    if show_plt:
        plt.show()
    else:
        if idx is None:
            plt.savefig(
                f'{save_dir}{saved_sampling_score_dir.split(".")[0]}_histogram_layer{layer_to_visualize}.png')
        else:
            plt.savefig(
                f'{save_dir}{saved_sampling_score_dir.split(".")[0]}_histogram_layer{layer_to_visualize}_sample{idx}.png')

    plt.close()


def sampling_score_bin_boundary(saved_sampling_score_dir='modelnet_sampling_scores.pt', layer_to_visualize=0):
    tensor = torch.load(saved_sampling_score_dir)
    num_batch = tensor.shape[0]
    print(tensor.shape)
    tensor = tensor[:, layer_to_visualize + 3, :]
    print(f'layer_to_visualize:{layer_to_visualize}')
    if layer_to_visualize != 0:
        tensor = torch.reshape(tensor, (-1,))
        tensor = tensor[tensor > -1.5]
        tensor = torch.reshape(tensor, (num_batch, 1024))

    tensor = (tensor - torch.mean(tensor, dim=1,
                                  keepdim=True))  # / torch.std(tensor, dim=1, unbiased=False, keepdim=True)
    flattened_tensor = tensor.flatten().cpu()
    # flattened_tensor = flattened_tensor[flattened_tensor > -1.5]
    sorted_value, sorted_index = torch.sort(flattened_tensor, dim=0)

    num_bins = 6
    for percent in range(0, num_bins + 1):
        percent = percent * 100 / num_bins
        if percent == 100:
            i = -1
        else:
            i = int(percent / 100 * len(sorted_value))
        print(f'{percent}% highest value is {sorted_value[i]}')


def visualization_sampling_score_in_bin(saved_sampling_score_dir='modelnet_sampling_scores.pt', layer_to_visualize=0,
                                        show_plt=False, save_dir=r'C:/Users/Lenovo/Desktop/SamplingScore/', idx=None,
                                        bin_boundary=[0.3, 0.5, 0.8, 1.3]):
    tensor = torch.load(saved_sampling_score_dir)
    num_batch = tensor.shape[0]
    print(tensor.shape)

    if idx is None:
        tensor = tensor[:, layer_to_visualize + 3, :]
    else:
        tensor = tensor[idx, layer_to_visualize + 3, :].reshape(1, -1)

    # Example tensor of shape [2464, 5, 2048]

    # Flatten the tensor to 1D for histogram

    print(f'layer_to_visualize:{layer_to_visualize}')
    if layer_to_visualize != 0:
        tensor = torch.reshape(tensor, (-1,))
        tensor = tensor[tensor > -1.5]
        if idx is None:
            tensor = torch.reshape(tensor, (num_batch, 1024))
        else:
            tensor = torch.reshape(tensor, (1, 1024))

    # tensor = (tensor - torch.min(tensor, dim=1, keepdim=True)[0]) / (torch.max(tensor, dim=1, keepdim=True)[0] - torch.min(tensor, dim=1, keepdim=True)[0] + 1e-8)
    # tensor = tensor / torch.std(tensor, dim=1, unbiased=False, keepdim=True)
    tensor = (tensor - torch.mean(tensor, dim=1,
                                  keepdim=True))  # / torch.std(tensor, dim=1, unbiased=False, keepdim=True)

    flattened_tensor = tensor.flatten().cpu()

    # if idx is None:
    #     # sorted_value = flattened_tensor[flattened_tensor > 0]
    #     # flattened_tensor = flattened_tensor[flattened_tensor > -1.5]
    #     # sorted_value, sorted_index = torch.sort(sorted_value, dim=0)
    #
    #     # for percent in range(0, 120, 20):
    #     #     if percent == 100:
    #     #         i = -1
    #     #     else:
    #     #         i = int(percent / 100 * len(sorted_value))
    #     #     print(f'{percent}% highest value is {sorted_value[i]}')
    #
    #     topk_values, _ = torch.topk(flattened_tensor, int(flattened_tensor.shape[0] * 0.03), largest=True)
    #     max_value_97 = topk_values[-1].item()
    #
    #     plt.figure()
    #     _, bins, patches = plt.hist(flattened_tensor, bins=500, range=(-0.5, max_value_97))  # (-2, 4))
    #
    #     colors = ['blue', 'cyan', 'green', 'red', 'magenta', 'yellow']
    #
    #     bin_boundary_extended = [-100]
    #     bin_boundary_extended.extend(bin_boundary)
    #
    #     for i in range(6):
    #         for bin_left, patch in zip(bins, patches):
    #             if bin_left > bin_boundary_extended[i]:
    #                 patch.set_facecolor(colors[i])
    # else:
    bin_boundary_extended = [-2]
    bin_boundary_extended.extend(bin_boundary)
    bin_boundary_extended.extend([100])

    hist, bin_edges = np.histogram(flattened_tensor, bins=bin_boundary_extended)
    plt.figure()
    plt.bar(np.arange(len(hist)), hist)

    if show_plt:
        plt.show()
    else:
        if idx is None:
            plt.savefig(
                f'{save_dir}{saved_sampling_score_dir.split(".")[0]}_layer{layer_to_visualize}.png')
        else:
            if layer_to_visualize == 1:
                plt.ylim((0, 550))
            else:
                plt.ylim((0, 750))
            plt.savefig(
                f'{save_dir}{saved_sampling_score_dir.split(".")[0]}_layer{layer_to_visualize}_sample{idx}.png')

    plt.close()

    # min_value = float(torch.min(flattened_tensor))
    #
    # topk_values, _ = torch.topk(flattened_tensor, int(flattened_tensor.shape[0] * 0.003), largest=True)
    # max_value_9772 = topk_values[-1].item()

    # max_value = float(np.max(flattened_tensor))

    # hist=torch.histc(flattened_tensor,bins=1000,min=min_value,max=max_value)
    # print(hist.)

    # Plotting the histogram

    # plt.figure()
    # plt.hist(flattened_tensor, bins=500, range=(min_value, max_value_9772))  # (-2, 4))
    # plt.title("Histogram of Tensor Elements")
    # plt.xlabel("Value")
    # plt.ylabel("Frequency")


def draw_pcd(idx, saved_sampling_score_dir='modelnet_sampling_scores.pt'):
    import torch
    import open3d as o3d

    tensor = torch.load(saved_sampling_score_dir)
    num_batch = tensor.shape[0]

    tensor = tensor[idx, 0:3, :].cpu().numpy().transpose(1, 0)
    print(tensor.shape)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(tensor)

    # o3d.io.write_point_cloud(save_dir, point_cloud, write_ascii=True)

    o3d.visualization.draw_geometries([point_cloud])


def normalization_in_bins():
    import pickle
    import torch

    with open(r'E:\datasets\APES\test_results\masked_attention_score.pkl', 'rb') as f:
        data_dict = pickle.load(f)

    masked_attention_score = data_dict['masked_attention_score']

    print(f'masked_attention_score.shape:{masked_attention_score.shape}')

    masked_attention_score_oneshape = masked_attention_score[1, 0, :, :].permute(1, 0)
    print(masked_attention_score_oneshape.permute(1, 0))

    print(f'masked_attention_score_oneshape.shape:{masked_attention_score_oneshape.shape}')

    print('----------------------------------------')
    print('No Fuhrther Process')
    max_value = []
    min_value = []
    mean_value = []
    std_value = []
    for bin_index in range(masked_attention_score_oneshape.shape[0]):
        masked_attention_score_oneshape_onebin = masked_attention_score_oneshape[bin_index, :]
        masked_attention_score_oneshape_onebin = masked_attention_score_oneshape_onebin[
            masked_attention_score_oneshape_onebin != 0]
        max_value.append(torch.max(masked_attention_score_oneshape_onebin).item())
        min_value.append((torch.min(masked_attention_score_oneshape_onebin).item()))
        mean_value.append((torch.mean(masked_attention_score_oneshape_onebin).item()))
        std_value.append((torch.std(masked_attention_score_oneshape_onebin).item()))
    print(f'max:{max_value}')
    print(f'min:{min_value}')
    print(f'mean:{mean_value}')
    print(f'std:{std_value}')

    print('----------------------------------------')
    print('Softmax')
    max_value = []
    min_value = []
    mean_value = []
    std_value = []
    for bin_index in range(masked_attention_score_oneshape.shape[0]):
        masked_attention_score_oneshape_onebin = masked_attention_score_oneshape[bin_index, :]
        Proposed = masked_attention_score_oneshape_onebin[
            masked_attention_score_oneshape_onebin != 0]

        Proposed = torch.nn.functional.softmax(Proposed, dim=0)

        max_value.append(torch.max(Proposed).item())
        min_value.append((torch.min(Proposed).item()))
        mean_value.append((torch.mean(Proposed).item()))
        std_value.append((torch.std(Proposed).item()))
    print(f'max:{max_value}')
    print(f'min:{min_value}')
    print(f'mean:{mean_value}')
    print(f'std:{std_value}')

    print('----------------------------------------')
    print('sigmoid and Softmax')
    max_value = []
    min_value = []
    mean_value = []
    std_value = []
    for bin_index in range(masked_attention_score_oneshape.shape[0]):
        masked_attention_score_oneshape_onebin = masked_attention_score_oneshape[bin_index, :]
        Proposed = masked_attention_score_oneshape_onebin[
            masked_attention_score_oneshape_onebin != 0]

        Proposed = torch.nn.functional.sigmoid(Proposed)
        Proposed = torch.nn.functional.softmax(Proposed, dim=0)

        max_value.append(torch.max(Proposed).item())
        min_value.append((torch.min(Proposed).item()))
        mean_value.append((torch.mean(Proposed).item()))
        std_value.append((torch.std(Proposed).item()))
    print(f'max:{max_value}')
    print(f'min:{min_value}')
    print(f'mean:{mean_value}')
    print(f'std:{std_value}')

    print('----------------------------------------')
    print('tanh and Softmax')
    max_value = []
    min_value = []
    mean_value = []
    std_value = []
    for bin_index in range(masked_attention_score_oneshape.shape[0]):
        masked_attention_score_oneshape_onebin = masked_attention_score_oneshape[bin_index, :]
        Proposed = masked_attention_score_oneshape_onebin[
            masked_attention_score_oneshape_onebin != 0]

        Proposed = torch.nn.functional.tanh(Proposed)
        Proposed = torch.nn.functional.softmax(Proposed, dim=0)

        max_value.append(torch.max(Proposed).item())
        min_value.append((torch.min(Proposed).item()))
        mean_value.append((torch.mean(Proposed).item()))
        std_value.append((torch.std(Proposed).item()))
    print(f'max:{max_value}')
    print(f'min:{min_value}')
    print(f'mean:{mean_value}')
    print(f'std:{std_value}')

    # print('----------------------------------------')
    # print('sqrt and Softmax')
    # max_value = []
    # min_value = []
    # mean_value = []
    # std_value = []
    # for bin_index in range(masked_attention_score_oneshape.shape[0]):
    #     masked_attention_score_oneshape_onebin = masked_attention_score_oneshape[bin_index, :]
    #     Proposed = masked_attention_score_oneshape_onebin[
    #         masked_attention_score_oneshape_onebin != 0]
    #
    #     Proposed = torch.sign(Proposed) * torch.pow(torch.abs(Proposed), 1 / 2) * 2
    #     Proposed = torch.nn.functional.softmax(Proposed, dim=0)
    #
    #     max_value.append(torch.max(Proposed).item())
    #     min_value.append((torch.min(Proposed).item()))
    #     mean_value.append((torch.mean(Proposed).item()))
    #     std_value.append((torch.std(Proposed).item()))
    # print(f'max:{max_value}')
    # print(f'min:{min_value}')
    # print(f'mean:{mean_value}')
    # print(f'std:{std_value}')
    #
    # print('----------------------------------------')
    # print('ln+1 and Softmax')
    # max_value = []
    # min_value = []
    # mean_value = []
    # std_value = []
    # for bin_index in range(masked_attention_score_oneshape.shape[0]):
    #     masked_attention_score_oneshape_onebin = masked_attention_score_oneshape[bin_index, :]
    #     Proposed = masked_attention_score_oneshape_onebin[
    #         masked_attention_score_oneshape_onebin != 0]
    #
    #     Proposed = torch.log(Proposed + 1)
    #     Proposed = torch.nn.functional.softmax(Proposed, dim=0)
    #
    #     max_value.append(torch.max(Proposed).item())
    #     min_value.append((torch.min(Proposed).item()))
    #     mean_value.append((torch.mean(Proposed).item()))
    #     std_value.append((torch.std(Proposed).item()))
    # print(f'max:{max_value}')
    # print(f'min:{min_value}')
    # print(f'mean:{mean_value}')
    # print(f'std:{std_value}')

    print('----------------------------------------')
    print('tanh and T0.3 and Softmax')
    max_value = []
    min_value = []
    mean_value = []
    std_value = []
    for bin_index in range(masked_attention_score_oneshape.shape[0]):
        masked_attention_score_oneshape_onebin = masked_attention_score_oneshape[bin_index, :]
        Proposed = masked_attention_score_oneshape_onebin[
            masked_attention_score_oneshape_onebin != 0]

        Proposed = torch.nn.functional.tanh(Proposed) / 0.3
        Proposed = torch.nn.functional.softmax(Proposed, dim=0)

        max_value.append(torch.max(Proposed).item())
        min_value.append((torch.min(Proposed).item()))
        mean_value.append((torch.mean(Proposed).item()))
        std_value.append((torch.std(Proposed).item()))
    print(f'max:{max_value}')
    print(f'min:{min_value}')
    print(f'mean:{mean_value}')
    print(f'std:{std_value}')

    print('----------------------------------------')
    print('tanh and T0.1 and Softmax')
    max_value = []
    min_value = []
    mean_value = []
    std_value = []
    for bin_index in range(masked_attention_score_oneshape.shape[0]):
        masked_attention_score_oneshape_onebin = masked_attention_score_oneshape[bin_index, :]
        Proposed = masked_attention_score_oneshape_onebin[
            masked_attention_score_oneshape_onebin != 0]

        Proposed = torch.nn.functional.tanh(Proposed) / 0.1
        Proposed = torch.nn.functional.softmax(Proposed, dim=0)

        max_value.append(torch.max(Proposed).item())
        min_value.append((torch.min(Proposed).item()))
        mean_value.append((torch.mean(Proposed).item()))
        std_value.append((torch.std(Proposed).item()))
    print(f'max:{max_value}')
    print(f'min:{min_value}')
    print(f'mean:{mean_value}')
    print(f'std:{std_value}')

    print('----------------------------------------')
    print('tanh and T0.05 and Softmax')
    max_value = []
    min_value = []
    mean_value = []
    std_value = []
    for bin_index in range(masked_attention_score_oneshape.shape[0]):
        masked_attention_score_oneshape_onebin = masked_attention_score_oneshape[bin_index, :]
        Proposed = masked_attention_score_oneshape_onebin[
            masked_attention_score_oneshape_onebin != 0]

        Proposed = torch.nn.functional.tanh(Proposed) / 0.05
        Proposed = torch.nn.functional.softmax(Proposed, dim=0)

        max_value.append(torch.max(Proposed).item())
        min_value.append((torch.min(Proposed).item()))
        mean_value.append((torch.mean(Proposed).item()))
        std_value.append((torch.std(Proposed).item()))
    print(f'max:{max_value}')
    print(f'min:{min_value}')
    print(f'mean:{mean_value}')
    print(f'std:{std_value}')
