import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d

from PIL import Image
from scipy.spatial.transform import Rotation as R

from tools.utils import label2color, visualize_depth_numpy
from .occmap3d import OccpancyGridMap3D


def convert_matrixs_to_vector(all_trajectory):
    poses = []

    for trajectory in all_trajectory:
        r_matrix = trajectory[:3, :3]
        tx, ty, tz = trajectory[:3, 3]

        rotation = R.from_matrix(r_matrix)
        qx, qy, qz, qw = rotation.as_quat()

        pose = [tx, ty, tz, qx, qy, qz, qw]
        poses.append(pose)

    poses = np.array(poses)
    return poses


class ExplicitMapper:
    def __init__(self, args, device):
        self.model = OccpancyGridMap3D()

        self.all_rays = []
        self.colors = []
        self.depths = []
        self.semantics = []
        self.extrinsics = []

        # self.frame_id = 0

    def update_map(self, iteration):
        rays = self.all_rays[-1]
        color = self.colors[-1]
        depth = self.depths[-1]
        semantic = self.semantics[-1]

        rays = rays.to(self.model.device)
        color = color.to(self.model.device)
        depth = depth.to(self.model.device)
        semantic = semantic.to(self.model.device)

        self.model.update(rays, color, depth, semantic)

        output = None
        iteration = 1

        return output, iteration

    def render_view(
        self,
        pose,
        intrinsics,
        H,
        W,
        downscale,
        n_samples,
        rendering_target_id=None,
        planning=False,
    ):
        extrinsic: np.ndarray = np.linalg.inv(pose.astype(np.float64))
        intrinsic = np.eye(3)
        intrinsic[0, 0] = intrinsics[0]
        intrinsic[1, 1] = intrinsics[1]
        intrinsic[0, 2] = intrinsics[2]
        intrinsic[1, 2] = intrinsics[3]
        img_h, img_w = H, W

        downscale = 1.0
        img_h = int(img_h * downscale)
        img_w = int(img_w * downscale)
        intrinsic[0] *= downscale
        intrinsic[1] *= downscale

        color_render = np.zeros((img_h, img_w, 3), dtype=np.float32)
        depth_render = np.zeros((img_h, img_w), dtype=np.float32)
        semantic_render = np.zeros((img_h, img_w, 3), dtype=np.int32)

        output = {
            "rgb": color_render,
            "semantic": semantic_render,
            "depth": depth_render,
            "depth_uncertainty": depth_render,
            "semantic_uncertainty": semantic_render,
            "acc": semantic_render,
        }

        return output

    def update_measurement_buffer(self, buffer, use_replay_buffer=True):
        (
            all_rays,
            all_rgbs,
            all_semantics,
            all_semantics_unc,
            all_depths,
            all_semantics_mask,
            all_semantics_gt,
            all_trajectory,
            intrinsics,
        ) = buffer

        img_h, img_w, focal, c = intrinsics

        self.intrinsics = np.eye(3)
        self.intrinsics[0, 0] = focal[0]
        self.intrinsics[1, 1] = focal[1]
        self.intrinsics[0, 2] = c[0]
        self.intrinsics[1, 2] = c[1]
        self.img_h, self.img_w = img_h, img_w

        frame_idx = all_trajectory.shape[0] - 1
        ray_index = img_h * img_w * frame_idx

        rays = all_rays[ray_index:]
        color = all_rgbs[ray_index:]
        depth = all_depths[ray_index:].view(-1, 1)
        semantic = all_semantics_gt[ray_index:]
        semantic = F.one_hot(
            semantic.long(), num_classes=self.model.num_semantic_classes
        ).float()
        extrinsic = np.linalg.inv(all_trajectory[-1])

        self.all_rays.append(rays)
        self.colors.append(color)
        self.depths.append(depth)
        self.semantics.append(semantic)
        self.extrinsics.append(extrinsic)

    def save_model(self, model_file):
        pass

    def save_mesh(self, mesh_file, target_class_id):
        self.model.save_mesh(mesh_file, target_class_id)
