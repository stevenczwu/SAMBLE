from pyntcloud import PyntCloud
from plyfile import PlyData, PlyElement
import numpy as np
import pickle
import argparse
import os
import pkbar
import torch
from pytorch3d.ops import sample_farthest_points as fps
import h5py
import glob


# def edge_sampling(input_path, output_path, ds_points, k=16, keep_tmp=False):
#
#     tmp_pts_path = f'{output_path}/tmp_pts'
#     tmp_labels_path = f'{output_path}/tmp_labels'
#     ds_path = f'{output_path}/ds{ds_points}'
#     output_dat_path = f'{output_path}/{os.path.basename(input_path).replace("1024", str(ds_points))}'
#     if not os.path.exists(tmp_pts_path):
#         os.makedirs(tmp_pts_path, exist_ok=True)
#     if not os.path.exists(tmp_labels_path):
#         os.makedirs(tmp_labels_path, exist_ok=True)
#     if not os.path.exists(ds_path):
#         os.makedirs(ds_path, exist_ok=True)
#
#     with open(input_path, 'rb') as f:
#         all_pcd, _ = pickle.load(f)
#     all_pcd = np.stack(all_pcd, axis=0)[:, :, :3]
#     all_pcd = torch.Tensor(all_pcd).cuda()
#     all_pcd, _ = fps(all_pcd, K=2*ds_points, random_start_point=True)
#     all_pcd = all_pcd.cpu().numpy()
#     pbar = pkbar.Pbar(name='Generating tmp points files, please wait...', target=all_pcd.shape[0])
#     for i, pcd in enumerate(all_pcd):
#         file_path = f'{tmp_pts_path}/{i}.ply'
#         pcd = pcd.tolist()
#         tmp = []
#         for point in pcd:
#             tmp.append(tuple(point))
#         vertex = PlyElement.describe(np.array(tmp, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]), 'vertex')
#         PlyData([vertex]).write(file_path)
#         pbar.update(i)
#
#     with open(input_path, 'rb') as f:
#         _, all_cls_label = pickle.load(f)
#     all_cls_label = np.stack(all_cls_label, axis=0)[:, 0]
#     pbar = pkbar.Pbar(name='Generating tmp label files, please wait...', target=all_cls_label.shape[0])
#     for i, label in enumerate(all_cls_label):
#         file_path = f'{tmp_labels_path}/{i}.txt'
#         with open(file_path, 'w') as f:
#             f.write(str(label))
#         pbar.update(i)
#
#     pbar = pkbar.Pbar(name='Generating downsampled files, please wait...', target=len(os.listdir(tmp_pts_path)))
#     for i, each_file in enumerate(os.listdir(tmp_pts_path)):
#         cloud = PyntCloud.from_file(f'{tmp_pts_path}/{each_file}')
#         k_neighbors = cloud.get_neighbors(k)
#         ev = cloud.add_scalar_field('eigen_values', k_neighbors=k_neighbors)
#         cloud.add_scalar_field('curvature', ev=ev)
#         idx = cloud.points['curvature(17)'].nlargest(n=ds_points).index
#         cloud.points = cloud.points.iloc[idx]
#         cloud.to_file(f'{ds_path}/{each_file}')
#         pbar.update(i)
#
#     points_list = []
#     label_list = []
#     pbar = pkbar.Pbar(name='Generating .dat files, please wait...', target=len(os.listdir(tmp_pts_path)))
#     for i, (pts, labels) in enumerate(zip(sorted(os.listdir(ds_path)), sorted(os.listdir(tmp_labels_path)))):
#         pts = PyntCloud.from_file(f'{ds_path}/{pts}').points
#         points_list.append(pts.to_numpy())
#         label_list.append(np.array([np.loadtxt(f'{tmp_labels_path}/{labels}')], dtype='int32'))
#         pbar.update(i)
#     with open(output_dat_path, 'wb') as f:
#         pickle.dump([points_list, label_list], f)
#
#     if not keep_tmp:
#         print('Removing tmp files, please wait...')
#         os.system(f'rm -rf {tmp_pts_path}')
#         os.system(f'rm -rf {tmp_labels_path}')
#         os.system(f'rm -rf {ds_path}')
#         print('Done!')


def edge_sampling(input_path, output_path, ds_points, k=16, keep_tmp=False):

    mode = ['train', 'test']

    for m in mode:
        tmp_pts_path = f'{output_path}/{m}_tmp_pts'
        tmp_labels_path = f'{output_path}/{m}_tmp_labels'
        ds_path = f'{output_path}/{m}_ds{ds_points}'
        output_dat_path = f'{output_path}/{m}_ds{ds_points}.dat'
        if not os.path.exists(tmp_pts_path):
            os.makedirs(tmp_pts_path, exist_ok=True)
        if not os.path.exists(tmp_labels_path):
            os.makedirs(tmp_labels_path, exist_ok=True)
        if not os.path.exists(ds_path):
            os.makedirs(ds_path, exist_ok=True)

        all_pcd = []
        all_cls_label = []
        file = glob.glob(os.path.join(input_path, f'*{m}*.h5'))
        for h5_name in file:
            f = h5py.File(h5_name, 'r+')
            pcd = f['data'][:].astype('float32')
            cls_label = f['label'][:].astype('int64')
            f.close()
            all_pcd.append(pcd)
            all_cls_label.append(cls_label[:, 0])
        all_pcd = np.concatenate(all_pcd, axis=0)
        all_pcd = torch.Tensor(all_pcd).cuda()
        all_pcd, _ = fps(all_pcd, K=2*ds_points, random_start_point=True)
        all_pcd = all_pcd.cpu().numpy()
        pbar = pkbar.Pbar(name='Generating tmp points files, please wait...', target=all_pcd.shape[0])
        for i, pcd in enumerate(all_pcd):
            file_path = f'{tmp_pts_path}/{i}.ply'
            pcd = pcd.tolist()
            tmp = []
            for point in pcd:
                tmp.append(tuple(point))
            vertex = PlyElement.describe(np.array(tmp, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]), 'vertex')
            PlyData([vertex]).write(file_path)
            pbar.update(i)

        all_cls_label = np.concatenate(all_cls_label, axis=0)
        pbar = pkbar.Pbar(name='Generating tmp label files, please wait...', target=all_cls_label.shape[0])
        for i, label in enumerate(all_cls_label):
            file_path = f'{tmp_labels_path}/{i}.txt'
            with open(file_path, 'w') as f:
                f.write(str(label))
            pbar.update(i)

        pbar = pkbar.Pbar(name='Generating downsampled files, please wait...', target=len(os.listdir(tmp_pts_path)))
        for i, each_file in enumerate(os.listdir(tmp_pts_path)):
            cloud = PyntCloud.from_file(f'{tmp_pts_path}/{each_file}')
            k_neighbors = cloud.get_neighbors(k)
            ev = cloud.add_scalar_field('eigen_values', k_neighbors=k_neighbors)
            cloud.add_scalar_field('curvature', ev=ev)
            idx = cloud.points['curvature(17)'].nlargest(n=ds_points).index
            cloud.points = cloud.points.iloc[idx]
            cloud.to_file(f'{ds_path}/{each_file}')
            pbar.update(i)

        points_list = []
        label_list = []
        pbar = pkbar.Pbar(name='Generating .dat files, please wait...', target=len(os.listdir(tmp_pts_path)))
        for i, (pts, labels) in enumerate(zip(sorted(os.listdir(ds_path)), sorted(os.listdir(tmp_labels_path)))):
            pts = PyntCloud.from_file(f'{ds_path}/{pts}').points
            points_list.append(pts.to_numpy())
            label_list.append(np.array([np.loadtxt(f'{tmp_labels_path}/{labels}')], dtype='int32'))
            pbar.update(i)
        with open(output_dat_path, 'wb') as f:
            pickle.dump([points_list, label_list], f)

        if not keep_tmp:
            print(f'Removing {m} tmp files, please wait...')
            os.system(f'rm -rf {tmp_pts_path}')
            os.system(f'rm -rf {tmp_labels_path}')
            os.system(f'rm -rf {ds_path}')
            print('Done!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', help='input folder of .h5 files')
    parser.add_argument('output_folder', help='output folder of the downsampled .dat file')
    parser.add_argument('ds_points', help='how many points you want to keep')
    parser.add_argument('-k', help='number of neighbor points', default=16)
    parser.add_argument('-keep_tmp_files', help='whether to keep tmp files', action='store_true')
    args = parser.parse_args()

    edge_sampling(args.input_folder, args.output_folder, int(args.ds_points), int(args.k), args.keep_tmp_files)
