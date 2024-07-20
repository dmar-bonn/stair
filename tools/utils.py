import cv2, torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import scipy.signal
from packaging import version as pver
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import random
import plyfile
import skimage.measure
from kornia import create_meshgrid
import lpips
from torch import nn
import trimesh

# import mcubes

mse2psnr = lambda x: -10.0 * torch.log(x) / torch.log(torch.Tensor([10.0]))


def BatchRenderer(
    rays,
    model,
    require_class=[],
    chunk=1024,
    n_samples=-1,
    white_bg=True,
    is_train=False,
    device="cuda",
    use_depth=False,
):
    (
        rgb_maps,
        depth_maps,
        semantic_maps,
        depth_unc_maps,
        semantic_unc_maps,
        acc_maps,
        unknown_maps,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    n_rays_all = rays.shape[0]
    for chunk_idx in range(n_rays_all // chunk + int(n_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
        (
            rgb_map,
            depth_map,
            semantic_map,
            depth_unc_map,
            semantic_unc_map,
            acc_map,
            unknown_map,
        ) = model(
            rays_chunk,
            require_class=require_class,
            is_train=is_train,
            white_bg=white_bg,
            n_samples=n_samples,
            use_depth=use_depth,
        )
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        semantic_maps.append(semantic_map)
        depth_unc_maps.append(depth_unc_map)
        semantic_unc_maps.append(semantic_unc_map)
        acc_maps.append(acc_map)
        unknown_maps.append(unknown_map)

    return (
        torch.cat(rgb_maps),
        torch.cat(semantic_maps),
        torch.cat(depth_maps),
        torch.cat(depth_unc_maps),
        torch.cat(semantic_unc_maps),
        torch.cat(acc_maps),
        torch.cat(unknown_maps),
    )


class SimpleSampler:
    def __init__(self, total, batch, offset=0):
        self.total = total
        self.batch = batch
        self.curr = total
        self.offset = offset
        self.ids = None

    def nextids(
        self,
    ):
        self.curr += self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr : self.curr + self.batch] + self.offset


class ReplaySampler:
    def __init__(self, active_buffer, total_new, batch, percent, offset=0):
        self.active_buffer = active_buffer
        self.total_buffer = len(active_buffer)
        self.curr_buffer = self.total_buffer
        self.total_new = total_new
        self.curr_new = total_new

        self.batch_buffer = int(batch * percent)
        self.batch_new = batch - self.batch_buffer
        self.offset = offset
        self.ids_buffer = None
        self.ids_new = None

    def nextids(self):
        self.curr_new += self.batch_new
        if self.curr_new + self.batch_new > self.total_new:
            self.ids_new = torch.LongTensor(np.random.permutation(self.total_new))
            self.curr_new = 0
        samples_all = (
            self.ids_new[self.curr_new : self.curr_new + self.batch_new] + self.offset
        )

        if self.total_buffer > 0:
            self.curr_buffer += self.batch_buffer
            if self.curr_buffer + self.batch_buffer > self.total_buffer:
                self.ids_buffer = torch.LongTensor(
                    np.random.permutation(self.total_buffer)
                )
                self.curr_buffer = 0
            samples_buffer = self.active_buffer[
                self.ids_buffer[self.curr_buffer : self.curr_buffer + self.batch_buffer]
            ]
            samples_all = torch.cat((samples_all, samples_buffer), dim=-1)

        return samples_all


LABELS = {
    "shapenet": {
        "background": {"color": (255, 255, 255), "id": 0},
        "car": {"color": (102, 102, 102), "id": 1},
        "chair": {"color": (0, 0, 255), "id": 2},
        "table": {"color": (0, 255, 255), "id": 3},
        "sofa": {"color": (255, 0, 0), "id": 4},
        "airplane": {"color": (102, 0, 204), "id": 5},
        "camera": {"color": (0, 102, 0), "id": 6},
    }
}


def label2onehot(label_map, class_num):
    label_map = torch.from_numpy(label_map).long()
    onehot = torch.nn.functional.one_hot(label_map, num_classes=class_num)
    return onehot.numpy()


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


def visualize_error_numpy(error, minmax=[0.0, 1.0], cmap=cv2.COLORMAP_BONE):
    """
    depth: (H, W)
    """

    x = np.nan_to_num(error)  # change nan to 0
    if minmax is None:
        mi = np.min(x[x > 0])  # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi, ma = minmax

    x = (x - mi) / (ma - mi)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    x_ = (cv2.cvtColor(x_, cv2.COLOR_BGR2RGB) / 255.0).astype(np.float32)
    return x_


def visualize_depth_numpy(depth, minmax=[0.0, 1.0], cmap=cv2.COLORMAP_BONE):
    """
    depth: (H, W)
    """

    x = np.nan_to_num(depth)  # change nan to 0
    if minmax is None:
        mi = np.min(x[x > 0])  # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi, ma = minmax

    x = (x - mi) / (ma - mi)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    x_ = (cv2.cvtColor(x_, cv2.COLOR_BGR2RGB) / 255.0).astype(np.float32)
    return x_


def visualize_depth_unc_numpy(depth_unc, minmax=[0.0, 1.0], cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """

    x = np.nan_to_num(depth_unc)  # change nan to 0
    # x = 1 - np.exp(-depth_unc / 3)
    # x = np.nan_to_num(depth_unc)  # change nan to 0
    # if minmax is None:
    #     mi = np.min(x[x > 0])  # get minimum positive depth (ignore background)
    #     ma = np.max(x)
    # else:
    #     mi, ma = minmax

    # x = (x - mi) / (ma - mi)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    x_ = (cv2.cvtColor(x_, cv2.COLOR_BGR2RGB) / 255.0).astype(np.float32)
    return x_


def visualize_semantic_mask_numpy(x, reverse_color=True, cmap=cv2.COLORMAP_BONE):
    """
    depth: (H, W)
    """
    # minmax = None
    # x = np.nan_to_num(semantic_mask)  # change nan to 0
    # if minmax is None:
    #     mi = np.min(x)  # get minimum positive depth (ignore background)
    #     ma = np.max(x)
    # else:
    #     mi, ma = minmax

    # x = (x - mi) / (ma - mi)  # normalize to 0~1
    if reverse_color:
        x = np.abs(1 - x)

    x = (255 * x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    x_ = (cv2.cvtColor(x_, cv2.COLOR_BGR2RGB) / 255.0).astype(np.float32)
    return x_


def visualize_semantic_unc_numpy(
    semantic_unc, minmax=[0.0, 1.0], cmap=cv2.COLORMAP_JET
):
    """
    depth: (H, W)
    """
    # minmax = None
    x = np.nan_to_num(semantic_unc)  # change nan to 0
    if minmax is None:
        mi = np.min(x)  # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi, ma = minmax

    x = (x - mi) / (ma - mi)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    x_ = (cv2.cvtColor(x_, cv2.COLOR_BGR2RGB) / 255.0).astype(np.float32)
    return x_


def visualize_acc_numpy(acc, minmax=[0.0, 1.0], cmap=cv2.COLORMAP_BONE):
    """
    depth: (H, W)
    """
    # minmax = None
    x = np.nan_to_num(acc)  # change nan to 0
    if minmax is None:
        mi = np.min(x)  # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi, ma = minmax

    x = (x - mi) / (ma - mi)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    x_ = (cv2.cvtColor(x_, cv2.COLOR_BGR2RGB) / 255.0).astype(np.float32)
    return x_


def init_log(log, keys):
    for key in keys:
        log[key] = torch.tensor([0.0], dtype=float)
    return log


def visualize_depth(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    if type(depth) is not np.ndarray:
        depth = depth.cpu().numpy()

    x = np.nan_to_num(depth)  # change nan to 0
    if minmax is None:
        mi = np.min(x[x > 0])  # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi, ma = minmax

    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_, [mi, ma]


def n_to_reso(n_voxels, bbox):
    xyz_min, xyz_max = bbox
    dim = len(xyz_min)
    voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / dim)
    return ((xyz_max - xyz_min) / voxel_size).long().tolist()


def cal_n_samples(reso, step_ratio=0.5):
    return int(np.linalg.norm(reso) / step_ratio)


def overlap_images(img1, img2):
    H, W, _ = img1.shape
    resolution = [int(H / 4), int(W / 4)]
    for i in range(len(img2)):
        img2[i] = cv2.resize(img2[i], dsize=resolution, interpolation=cv2.INTER_AREA)

    output = np.hstack((*img2,)).astype(np.float32)
    dim = output.shape
    img1[-dim[0] :, -dim[1] :] = output

    return img1


class ViewPyramid:
    def __init__(
        self,
    ):
        self.default_vertices = np.array(
            [
                [-0.1, -0.1, 0.3],
                [-0.1, 0.1, 0.3],
                [0.1, 0.1, 0.3],
                [0.1, -0.1, 0.3],
                [0, 0, 0],
            ]
        )

    def vertices(self, camera_pose):
        rotation = camera_pose[:3, :3]
        translation = camera_pose[:3, 3]
        new_v = translation + np.einsum("ij,kj->ik", self.default_vertices, rotation)
        return new_v


class TrajectoryVisualizer:
    def __init__(self, H, W):
        px = 1 / plt.rcParams["figure.dpi"]
        self.fig = plt.figure(figsize=(px * W, px * H))
        self.ax = self.fig.add_subplot(projection="3d")
        # self.ax.set_proj_type("persp", focal_length=0.3)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.xaxis.set_ticklabels([])
        self.ax.yaxis.set_ticklabels([])
        self.ax.zaxis.set_ticklabels([])
        self.ax.axes.set_xlim3d(left=-3, right=3)
        self.ax.axes.set_ylim3d(bottom=-3, top=3)
        self.ax.axes.set_zlim3d(bottom=0, top=3)
        self.ax.set_proj_type("persp")
        # self.ax.set_box_aspect(aspect=(1, 1, 1))
        self.plot_origin()
        self.view_pyramid = ViewPyramid()
        self.point_list = []
        self.measurement_num = 0

    def plot_origin(self):
        u, v = np.mgrid[0 : 2 * np.pi : 50j, 0 : np.pi : 50j]
        x = 0.05 * np.cos(u) * np.sin(v)
        y = 0.05 * np.sin(u) * np.sin(v)
        z = 0.05 * np.cos(v)
        self.ax.plot_surface(x, y, z, color="black", alpha=1)

    def add_view(self, pose):
        v = self.view_pyramid.vertices(pose)
        verts = [
            [v[0], v[1], v[4]],
            [v[0], v[3], v[4]],
            [v[2], v[1], v[4]],
            [v[2], v[3], v[4]],
        ]
        self.ax.scatter3D(v[:, 0], v[:, 1], v[:, 2], color="red")
        self.ax.add_collection3d(
            Poly3DCollection(
                verts, facecolors="cyan", linewidths=0.5, edgecolors="r", alpha=0.1
            )
        )
        point = v[-1]
        self.point_list.append(point)

        label = "[%.2f, %.2f, %.2f]" % (
            point[0],
            point[1],
            point[2],
        )
        self.ax.text(point[0], point[1], point[2], label, "x")

    # TODO: seem have bug here, camera jumps from time to time
    def pose2view(self, view_pose):
        xyz = view_pose[:3, 3]
        roll = 0
        if view_pose[2, 1] > 0:
            roll = 180
        if xyz[1] == 0:
            azim = -90
        else:
            azim = np.rad2deg(np.arctan2(xyz[1], xyz[0]))
        if xyz[0] + xyz[1] == 0:
            elev = 90
        else:
            elev = np.rad2deg(np.arctan(xyz[2] / np.sqrt(xyz[0] ** 2 + xyz[1] ** 2)))
        dist = np.sqrt(np.sum(xyz**2))
        return (elev, azim, roll, dist)

    def get_rendering(self, viewpose):
        elev, azim, roll, dist = self.pose2view(viewpose)
        self.ax.view_init(elev=elev, azim=azim)
        self.ax.dist = 4 * dist
        self.fig.canvas.draw()
        data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = (
            data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,)) / 255.0
        ).astype(np.float32)

        if roll == 180:
            data = cv2.rotate(data, cv2.ROTATE_180)

        return data


__LPIPS__ = {}


def init_lpips(net_name, device):
    assert net_name in ["alex", "vgg"]
    print(f"init_lpips: lpips_{net_name}")
    return lpips.LPIPS(net=net_name, version="0.1").eval().to(device)


def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()


def findItem(items, target):
    for one in items:
        if one[: len(target)] == target:
            return one
    return None


""" Evaluation metrics (ssim, lpips)
"""


def rgb_ssim(
    img0,
    img1,
    max_val,
    filter_size=11,
    filter_sigma=1.5,
    k1=0.01,
    k2=0.03,
    return_map=False,
):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma) ** 2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode="valid")

    filt_fn = lambda z: np.stack(
        [
            convolve2d(convolve2d(z[..., i], filt[:, None]), filt[None, :])
            for i in range(z.shape[-1])
        ],
        -1,
    )
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0.0, sigma00)
    sigma11 = np.maximum(0.0, sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class TVLoss3D(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss3D, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]  # B C H W D
        h_x = x.size()[2]
        w_x = x.size()[3]
        d_x = x.size()[4]
        count_h = self._tensor_size(x[:, :, 1:, :, :])
        count_w = self._tensor_size(x[:, :, :, 1:, :])
        count_d = self._tensor_size(x[:, :, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :, :] - x[:, :, : h_x - 1, :, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:, :] - x[:, :, :, : w_x - 1, :]), 2).sum()
        d_tv = torch.pow((x[:, :, :, :, 1:] - x[:, :, :, :, : d_x - 1]), 2).sum()
        return (
            self.TVLoss_weight
            * 2
            * (h_tv / count_h + w_tv / count_w + d_tv / count_d)
            / batch_size
        )

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def convert_occ_samples_to_ply(
    pytorch_3d_tensor,
    bbox,
    level=0.5,
    offset=None,
    scale=None,
):
    """
    Convert occ samples to .ply

    :param pytorch_3d_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_tensor = pytorch_3d_tensor.cpu().numpy()
    bbox = bbox.cpu().numpy()
    voxel_size = list((bbox[1] - bbox[0]) / np.array(pytorch_3d_tensor.shape))
    print("spacing", voxel_size)

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_tensor, level=level, spacing=voxel_size
    )
    faces = faces[..., ::-1]  # inverse face orientation

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = bbox[0, 0] + verts[:, 0]
    mesh_points[:, 1] = bbox[0, 1] + verts[:, 1]
    mesh_points[:, 2] = bbox[0, 2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    # print("saving mesh to %s" % (ply_filename_out))
    # ply_data.write(ply_filename_out)
    return ply_data


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse("1.10"):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing="ij")


def get_ray_directions(intrinsics):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the unnormalized direction of the rays in camera coordinate
    """
    H, W, focal, center = intrinsics
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0] + 0.5

    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    cent = center if center is not None else [W / 2, H / 2]
    directions = torch.stack(
        [(i - cent[0]) / focal[0], (j - cent[1]) / focal[1], torch.ones_like(i)], -1
    )  # (H, W, 3)

    return directions


def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T  # (H, W, 3)
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


def get_rays_gui(poses, intrinsics, H, W):
    device = poses.device
    B = poses.shape[0]  # (B, 4, 4)
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(
        torch.linspace(0, W - 1, W, device=device),
        torch.linspace(0, H - 1, H, device=device),
    )  # float
    i = i.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    j = j.t().reshape([1, H * W]).expand([B, H * W]) + 0.5

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    # directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)  # (B, H*W, 3)

    rays_o = poses[..., :3, 3]  # (B, 3)
    rays_o = rays_o[..., None, :].expand_as(rays_d)  # (B, H*W, 3)

    rays = torch.cat([rays_o, rays_d], 2).reshape(-1, 6)  # (B*H*W, 6)

    return rays


def to_mesh(occ):
    # Shorthand

    # Shape of voxel grid
    nx, ny, nz = occ.shape
    # Shape of corresponding occupancy grid
    grid_shape = (nx + 1, ny + 1, nz + 1)

    # Convert values to occupancies
    occ = np.pad(occ, 1, "constant")

    # Determine if face present
    f1_r = occ[:-1, 1:-1, 1:-1] & ~occ[1:, 1:-1, 1:-1]
    f2_r = occ[1:-1, :-1, 1:-1] & ~occ[1:-1, 1:, 1:-1]
    f3_r = occ[1:-1, 1:-1, :-1] & ~occ[1:-1, 1:-1, 1:]

    f1_l = ~occ[:-1, 1:-1, 1:-1] & occ[1:, 1:-1, 1:-1]
    f2_l = ~occ[1:-1, :-1, 1:-1] & occ[1:-1, 1:, 1:-1]
    f3_l = ~occ[1:-1, 1:-1, :-1] & occ[1:-1, 1:-1, 1:]

    f1 = f1_r | f1_l
    f2 = f2_r | f2_l
    f3 = f3_r | f3_l

    assert f1.shape == (nx + 1, ny, nz)
    assert f2.shape == (nx, ny + 1, nz)
    assert f3.shape == (nx, ny, nz + 1)

    # Determine if vertex present
    v = np.full(grid_shape, False)

    v[:, :-1, :-1] |= f1
    v[:, :-1, 1:] |= f1
    v[:, 1:, :-1] |= f1
    v[:, 1:, 1:] |= f1

    v[:-1, :, :-1] |= f2
    v[:-1, :, 1:] |= f2
    v[1:, :, :-1] |= f2
    v[1:, :, 1:] |= f2

    v[:-1, :-1, :] |= f3
    v[:-1, 1:, :] |= f3
    v[1:, :-1, :] |= f3
    v[1:, 1:, :] |= f3

    # Calculate indices for vertices
    n_vertices = v.sum()
    v_idx = np.full(grid_shape, -1)
    v_idx[v] = np.arange(n_vertices)

    # Vertices
    v_x, v_y, v_z = np.where(v)
    v_x = v_x / nx - 0.5
    v_y = v_y / ny - 0.5
    v_z = v_z / nz - 0.5
    vertices = np.stack([v_x, v_y, v_z], axis=1)

    # Face indices
    f1_l_x, f1_l_y, f1_l_z = np.where(f1_l)
    f2_l_x, f2_l_y, f2_l_z = np.where(f2_l)
    f3_l_x, f3_l_y, f3_l_z = np.where(f3_l)

    f1_r_x, f1_r_y, f1_r_z = np.where(f1_r)
    f2_r_x, f2_r_y, f2_r_z = np.where(f2_r)
    f3_r_x, f3_r_y, f3_r_z = np.where(f3_r)

    faces_1_l = np.stack(
        [
            v_idx[f1_l_x, f1_l_y, f1_l_z],
            v_idx[f1_l_x, f1_l_y, f1_l_z + 1],
            v_idx[f1_l_x, f1_l_y + 1, f1_l_z + 1],
            v_idx[f1_l_x, f1_l_y + 1, f1_l_z],
        ],
        axis=1,
    )

    faces_1_r = np.stack(
        [
            v_idx[f1_r_x, f1_r_y, f1_r_z],
            v_idx[f1_r_x, f1_r_y + 1, f1_r_z],
            v_idx[f1_r_x, f1_r_y + 1, f1_r_z + 1],
            v_idx[f1_r_x, f1_r_y, f1_r_z + 1],
        ],
        axis=1,
    )

    faces_2_l = np.stack(
        [
            v_idx[f2_l_x, f2_l_y, f2_l_z],
            v_idx[f2_l_x + 1, f2_l_y, f2_l_z],
            v_idx[f2_l_x + 1, f2_l_y, f2_l_z + 1],
            v_idx[f2_l_x, f2_l_y, f2_l_z + 1],
        ],
        axis=1,
    )

    faces_2_r = np.stack(
        [
            v_idx[f2_r_x, f2_r_y, f2_r_z],
            v_idx[f2_r_x, f2_r_y, f2_r_z + 1],
            v_idx[f2_r_x + 1, f2_r_y, f2_r_z + 1],
            v_idx[f2_r_x + 1, f2_r_y, f2_r_z],
        ],
        axis=1,
    )

    faces_3_l = np.stack(
        [
            v_idx[f3_l_x, f3_l_y, f3_l_z],
            v_idx[f3_l_x, f3_l_y + 1, f3_l_z],
            v_idx[f3_l_x + 1, f3_l_y + 1, f3_l_z],
            v_idx[f3_l_x + 1, f3_l_y, f3_l_z],
        ],
        axis=1,
    )

    faces_3_r = np.stack(
        [
            v_idx[f3_r_x, f3_r_y, f3_r_z],
            v_idx[f3_r_x + 1, f3_r_y, f3_r_z],
            v_idx[f3_r_x + 1, f3_r_y + 1, f3_r_z],
            v_idx[f3_r_x, f3_r_y + 1, f3_r_z],
        ],
        axis=1,
    )

    faces = np.concatenate(
        [
            faces_1_l,
            faces_1_r,
            faces_2_l,
            faces_2_r,
            faces_3_l,
            faces_3_r,
        ],
        axis=0,
    )

    mesh = trimesh.Trimesh(vertices, faces, process=False)
    return mesh


def cal_ause(err_vec, uncert_vec):
    # import matplotlib

    # matplotlib.use("TkAgg")
    # import matplotlib.pyplot as plt

    ratio_removed = np.linspace(0, 1, 100, endpoint=False)
    err_vec_sorted, _ = torch.sort(err_vec)

    # Calculate the error when removing a fraction pixels with error
    n_valid_pixels = len(err_vec)
    rmse_err = []
    for i, r in enumerate(ratio_removed):
        mse_err_slice = err_vec_sorted[0 : int((1 - r) * n_valid_pixels)]
        rmse_err.append(torch.sqrt(mse_err_slice).mean().cpu().numpy())

    # Normalize RMSE
    rmse_err = rmse_err / rmse_err[0]

    ###########################################

    # Sort by variance
    # print('Sorting Variance ...')
    uncert_vec = torch.sqrt(uncert_vec)
    _, uncert_vec_sorted_idxs = torch.sort(uncert_vec, descending=True)

    # Sort error by variance
    err_vec_sorted_by_uncert = err_vec[uncert_vec_sorted_idxs]

    rmse_err_by_var = []
    for i, r in enumerate(ratio_removed):
        mse_err_slice = err_vec_sorted_by_uncert[0 : int((1 - r) * n_valid_pixels)]
        rmse_err_by_var.append(torch.sqrt(mse_err_slice).mean().cpu().numpy())

    # Normalize RMSE
    rmse_err_by_var = rmse_err_by_var / max(rmse_err_by_var)

    # plt.plot(ratio_removed, rmse_err, "--")
    # plt.plot(ratio_removed, rmse_err_by_var, "-r")
    # plt.show()
    ause = np.trapz(np.array(rmse_err_by_var) - np.array(rmse_err), ratio_removed)
    return ause


def quat_to_rot(q):
    """
    Quaternion to rotation matrix
    """
    batch_size, _ = q.shape
    q = F.normalize(q, dim=1)
    R = torch.ones((batch_size, 3, 3), device=q.device)
    qr = q[:, 3]
    qi = q[:, 0]
    qj = q[:, 1]
    qk = q[:, 2]
    R[:, 0, 0] = 1 - 2 * (qj**2 + qk**2)
    R[:, 0, 1] = 2 * (qj * qi - qk * qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1 - 2 * (qi**2 + qk**2)
    R[:, 1, 2] = 2 * (qj * qk - qi * qr)
    R[:, 2, 0] = 2 * (qk * qi - qj * qr)
    R[:, 2, 1] = 2 * (qj * qk + qi * qr)
    R[:, 2, 2] = 1 - 2 * (qi**2 + qj**2)
    return R
