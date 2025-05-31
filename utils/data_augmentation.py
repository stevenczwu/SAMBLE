import numpy as np


def jitter(pcd, std=0.01, clip=0.05):
    num_points, num_features = pcd.shape  # pcd.shape == (N, 3)
    jittered_point = np.clip(
        std * np.random.randn(num_points, num_features), -clip, clip
    )
    jittered_point += pcd
    return jittered_point


def rotate(pcd, which_axis, angle_range):
    angle = np.random.uniform(angle_range[0], angle_range[1])
    angle = np.pi * angle / 180
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    if which_axis == "x":
        rotation_matrix = np.array(
            [[1, 0, 0], [0, cos_theta, sin_theta], [0, -sin_theta, cos_theta]]
        )
    elif which_axis == "y":
        rotation_matrix = np.array(
            [[cos_theta, 0, -sin_theta], [0, 1, 0], [sin_theta, 0, cos_theta]]
        )
    elif which_axis == "z":
        rotation_matrix = np.array(
            [[cos_theta, sin_theta, 0], [-sin_theta, cos_theta, 0], [0, 0, 1]]
        )
    else:
        raise ValueError(
            f"which_axis should be one of x, y and z, but got {which_axis}!"
        )
    rotated_points = pcd @ rotation_matrix
    return rotated_points


def translate(pcd, x_range, y_range, z_range, normal_channel=False):
    num_points = pcd.shape[0]
    x_translation = np.random.uniform(x_range[0], x_range[1])
    y_translation = np.random.uniform(y_range[0], y_range[1])
    z_translation = np.random.uniform(z_range[0], z_range[1])
    x = np.full(num_points, x_translation)
    y = np.full(num_points, y_translation)
    z = np.full(num_points, z_translation)
    translation = np.stack([x, y, z], axis=-1)
    if normal_channel:
        xyz = pcd[:, :3] + translation
        normal = pcd[:, 3:]
        pcd = np.concatenate([xyz, normal], axis=-1)
    else:
        pcd = pcd + translation
    return pcd


def anisotropic_scale(
    pcd, x_range, y_range, z_range, isotropic=False, normal_channel=False
):
    x_factor = np.random.uniform(x_range[0], x_range[1])
    y_factor = np.random.uniform(y_range[0], y_range[1])
    z_factor = np.random.uniform(z_range[0], z_range[1])
    if isotropic:
        scale_matrix = np.array([[x_factor, 0, 0], [0, x_factor, 0], [0, 0, x_factor]])
    else:
        scale_matrix = np.array([[x_factor, 0, 0], [0, y_factor, 0], [0, 0, z_factor]])

    if normal_channel:
        assert isotropic == True, "Normal channel only support isotropic scaling!"
        xyz = pcd[:, :3]
        normal = pcd[:, 3:]
        scaled_points = xyz @ scale_matrix
        scaled_normal = normal
        pcd = np.concatenate([scaled_points, scaled_normal], axis=-1)
    else:
        pcd = pcd @ scale_matrix
    return pcd


def rotate_perturbation_with_normal(pcd_normal, std=0.06, clip=0.18):
    """Randomly perturb the point clouds by small rotations
    Input:
      Nx6 array, original batch of point clouds and point normals
    Return:
      Nx6 array, rotated batch of point clouds
    """
    assert pcd_normal.shape[-1] == 6, "Input point cloud must with normal!"
    rotated_data = np.zeros(pcd_normal.shape, dtype=np.float32)
    angles = np.clip(std * np.random.randn(3), -clip, clip)
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angles[0]), -np.sin(angles[0])],
            [0, np.sin(angles[0]), np.cos(angles[0])],
        ]
    )
    Ry = np.array(
        [
            [np.cos(angles[1]), 0, np.sin(angles[1])],
            [0, 1, 0],
            [-np.sin(angles[1]), 0, np.cos(angles[1])],
        ]
    )
    Rz = np.array(
        [
            [np.cos(angles[2]), -np.sin(angles[2]), 0],
            [np.sin(angles[2]), np.cos(angles[2]), 0],
            [0, 0, 1],
        ]
    )
    R = np.dot(Rz, np.dot(Ry, Rx))
    shape_pc = pcd_normal[:, 0:3]
    shape_normal = pcd_normal[:, 3:6]
    rotated_data[:, 0:3] = np.dot(shape_pc.reshape((-1, 3)), R)
    rotated_data[:, 3:6] = np.dot(shape_normal.reshape((-1, 3)), R)
    return rotated_data


def rotate_with_normal(pcd_normal, angle_range):
    """Rotate the point cloud along up direction with certain angle.
    Input:
      Nx6 array, original batch of point clouds with normal
      scalar, angle of rotation
    Return:
      Nx6 array, rotated batch of point clouds iwth normal
    """
    if angle_range == None:
        angle = np.random.uniform() * 2 * np.pi
    else:
        angle = np.random.uniform(angle_range[0], angle_range[1])

    angle = np.pi * angle / 180

    cosval = np.cos(angle)
    sinval = np.sin(angle)
    rotation_matrix = np.array([[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]])
    shape_pc = pcd_normal[:, 0:3]
    shape_normal = pcd_normal[:, 3:6]
    pcd_normal[:, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    pcd_normal[:, 3:6] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)
    return pcd_normal
