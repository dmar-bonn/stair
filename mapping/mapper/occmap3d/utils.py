import numpy as np
import torch
from matplotlib import pyplot as plt


def plot_voxels(voxels):
    """
    Plot the voxels in 3D space.

    Args:
    - voxels: A numpy array of shape (N, 3) representing the coordinates of N voxels.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each voxel as a point
    ax.scatter(voxels[:, 0], voxels[:, 1], voxels[:, 2], c='blue', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Voxel Visualization')
    plt.show()


def plot_rays(ray_origins, ray_end_points):
    """
    Plot the rays from their origins to end points.

    Args:
    - ray_origins: A numpy array of shape (N, 3) representing the origins of N rays.
    - ray_end_points: A numpy array of shape (N, 3) representing the end points of N rays.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    for origin, end in zip(ray_origins, ray_end_points):
        # Each ray is represented by a line from its origin to its end point
        ax.plot([origin[0], end[0]], [origin[1], end[1]], [origin[2], end[2]], 'r-')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Ray Visualization')
    plt.show()


def get_rays(depth, intrinsic, extrinsic):
    """
    Args:
        depth: (H, W)
        intrinsic: (3, 3)
        extrinsic: (4, 4)
    Returns:
        rays: (H*W, 6): ray_o + ray_d
    """
    H, W = depth.shape  # Height and Width of the depth image
    device = depth.device

    # Generate grid of coordinates (u, v) for each pixel in the depth image
    u, v = torch.meshgrid(
        torch.arange(W, device=device, dtype=torch.float),
        torch.arange(H, device=device, dtype=torch.float),
        indexing='xy'
    )
    uv_homogeneous = torch.stack((u.flatten(), v.flatten(), torch.ones_like(u).flatten()), dim=1).T

    # Inverse of intrinsic matrix to unproject points
    intrinsic_inv = torch.inverse(intrinsic)

    # Unproject to camera space
    cam_coords = intrinsic_inv @ uv_homogeneous  # Shape: [3, H*W]
    cam_coords *= depth.flatten().unsqueeze(0)  # Scale with depth values

    # Transform to world coordinates
    world_coords = extrinsic[:3, :3] @ cam_coords + extrinsic[:3, 3].unsqueeze(1)

    # Camera position in world coordinates (extracted from extrinsic matrix)
    cam_position = extrinsic[:3, 3]

    # Ray directions (normalized)
    ray_dirs = world_coords - cam_position.unsqueeze(1)
    ray_dirs /= torch.norm(ray_dirs, dim=0, keepdim=True)

    # Concatenate ray origin and ray direction
    rays = torch.cat([cam_position.unsqueeze(-1).repeat(1, H * W), ray_dirs], dim=0).T

    return rays


def world_to_voxel_coordinates(world_points, voxel_size, grid_origin):
    """
    Convert points from world coordinates to voxel grid coordinates using PyTorch.

    Args:
    - world_points: A PyTorch tensor of shape (N, 3) representing N points in world coordinates.
    - voxel_size: The size of a single voxel (assumed to be the same in all dimensions).
    - grid_origin: The world coordinate of the origin of the voxel grid.

    Returns:
    - voxel_points: A PyTorch tensor of shape (N, 3) representing the converted voxel grid coordinates.
    """
    # Subtract the grid origin from the world points and scale according to the voxel size
    translated_points = (world_points + grid_origin) / voxel_size

    # Convert to integer indices
    voxel_points = torch.floor(translated_points).to(torch.int32)

    return voxel_points


def bresenham_3d_fn(x0, y0, z0, x1, y1, z1):
    """
    Yield integer coordinates on the line from (x0, y0, z0) to (x1, y1, z1).

    Input coordinates should be integers.
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)
    xs = 1 if x1 > x0 else -1
    ys = 1 if y1 > y0 else -1
    zs = 1 if z1 > z0 else -1

    # Driving axis is X-axis
    if dx >= dy and dx >= dz:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while x0 != x1:
            x0 += xs
            if p1 >= 0:
                y0 += ys
                p1 -= 2 * dx
            if p2 >= 0:
                z0 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            yield x0, y0, z0

    # Driving axis is Y-axis
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y0 != y1:
            y0 += ys
            if p1 >= 0:
                x0 += xs
                p1 -= 2 * dy
            if p2 >= 0:
                z0 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            yield x0, y0, z0

    # Driving axis is Z-axis
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z0 != z1:
            z0 += zs
            if p1 >= 0:
                y0 += ys
                p1 -= 2 * dz
            if p2 >= 0:
                x0 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            yield x0, y0, z0


def bresenham_3d(tensor_origin, tensor_end):
    """
    Return a list of all voxels intersected by the ray.
    """
    # Extracting integer coordinates from tensors
    x0, y0, z0 = tensor_origin.int().tolist()
    x1, y1, z1 = tensor_end.int().tolist()

    # Generate voxels using Bresenham's 3D algorithm
    voxels = list(bresenham_3d_fn(x0, y0, z0, x1, y1, z1))
    return torch.tensor(voxels, dtype=torch.long, device=tensor_origin.device)


def prob2logodds(p):
    return torch.log(p / (1 - p))


def logodds2prob(l):
    return 1 - 1 / (1 + torch.exp(l))


# def inv_sensor_model(cell, endpoint, prob_occ, prob_free):
#     if np.array_equal(cell, endpoint):
#         p = prob_occ
#     else:
#         p = prob_free
#
#     return prob2logodds(p)


def inv_sensor_model(voxel, endpoint, prob_occ, prob_free):
    """
    Inverse sensor model using PyTorch.

    Args:
    - voxel: PyTorch tensor representing a voxel coordinate.
    - endpoint: PyTorch tensor representing the endpoint coordinate.
    - prob_occ: Probability of being occupied.
    - prob_free: Probability of being free.

    Returns:
    - Log-odds value for the cell.
    """
    p = torch.where(torch.all(torch.eq(voxel, endpoint), dim=-1), prob_occ, prob_free)

    return prob2logodds(p)


LABELS = {
    "shapenet": {
        "background": {"color": (255, 255, 255), "id": 0},
        "car": {"color": (102, 102, 102), "id": 1},
        "chair": {"color": (0, 0, 255), "id": 2},
        "table": {"color": (0, 255, 255), "id": 3},
        "sofa": {"color": (255, 0, 0), "id": 4},
        "airplane": {"color": (102, 0, 204), "id": 5},
        "camera": {"color": (0, 102, 0), "id": 6},
        # "birdhouse": {"color": (255, 153, 204), "id": 7},
    }
}


def label2color(label_map, theme="shapenet"):
    assert theme in LABELS.keys()
    rgb = np.zeros((label_map.shape[0], label_map.shape[1], 3), np.float32)
    for _, cl in LABELS[theme].items():  # loop each class label
        if cl["color"] == (0, 0, 0):
            continue  # skip assignment of only zeros
        mask = label_map == cl["id"]
        rgb[:, :, 0][mask] = cl["color"][0] / 255.0
        rgb[:, :, 1][mask] = cl["color"][1] / 255.0
        rgb[:, :, 2][mask] = cl["color"][2] / 255.0
    return rgb
