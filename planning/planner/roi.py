from scipy.spatial import distance
from ..planner_base import PlannerBase
from ..planning_utils import *
import cv2
from tools.utils import *
from tqdm import tqdm
import os


class ROIPlanner(PlannerBase):
    def __init__(self, args, device):
        super(ROIPlanner, self).__init__(args, device)
        self.num_candidates = args.num_candidates
        self.util_weight = args.util_weight
        self.downscale_planning = args.downscale_planning
        self.rendering_intrinsics = np.array([*self.focal, self.W // 2, self.H // 2])
        self.max_dist = torch.tensor([10]).to(self.device)
        self.offset = np.array([1.5, 1.5, 1.5])
        self.init_planning()

    def update(self, _):
        self.mapper.update_map(0)
        return None, 1

    def save_mesh(self, mesh_name="final"):
        print("------ save mesh ------\n")
        mesh_folder = f"{self.record_path}/meshes"
        os.makedirs(mesh_folder, exist_ok=True)
        mesh_file = f"{mesh_folder}/mesh_{mesh_name}.ply"

        if self.save_scene:
            target_id = None
        else:
            target_id = self.planning_target_id

        self.mapper.save_mesh(mesh_file, target_id)

    def plan_next_view(self):
        view_list = np.empty((self.num_candidates, 2))

        for i in range(self.num_candidates):
            view_list[i] = uniform_sampling(self.radius, self.phi_min)

        candidate_poses = view_to_pose_batch(view_list, self.radius)
        candidate_poses = gazebo2opencv(candidate_poses).astype(np.float32)
        candidate_poses[:, :3, 3] += self.offset

        candidate_poses = torch.from_numpy(candidate_poses).to(self.device)
        utility_list = self.get_candidate_utility(candidate_poses)
        nbv_indeces = np.argsort(utility_list)[::-1]
        return [view_list[nbv_indeces[0]]]

    def get_candidate_utility(self, candidate_poses):
        rH = int(self.H * self.downscale_planning)
        rW = int(self.W * self.downscale_planning)
        intrinsics = self.rendering_intrinsics * self.downscale_planning
        utility_list = np.zeros(self.num_candidates)
        roi_index = self.mapper.model.get_region_of_interest(self.planning_target_id)

        for i, candidate_pose in tqdm(
            zip(range(self.num_candidates), candidate_poses),
            desc="calculate view utility",
            total=self.num_candidates,
        ):
            # for i, candidate_pose in enumerate(candidate_poses):
            view_utility = 0
            rays = get_rays_gui(
                candidate_pose.unsqueeze(0), intrinsics, rH, rW
            )  # (1*H*W, 6)
            for ray in rays:
                voxel_index = self.mapper.model.search_voxels_along_ray(ray)  # (N ,3)
                point_order = torch.arange(len(voxel_index)).to(self.device)
                occupancy_state = self.mapper.model.voxel_occupancy[
                    voxel_index[:, 0], voxel_index[:, 1], voxel_index[:, 2]
                ]
                occupancy_mask = (occupancy_state > 0.9).view(-1)
                weight_state = self.mapper.model.voxel_weights[
                    voxel_index[:, 0], voxel_index[:, 1], voxel_index[:, 2]
                ]
                unknown_mask = (weight_state == 0).view(-1)

                if occupancy_mask.any():
                    occupancy_point_order = point_order[occupancy_mask]
                    # print(occupancy_point_order)
                    first_occupancy_point = torch.min(occupancy_point_order)
                    valid_mask = point_order < (first_occupancy_point + 1)
                    unknown_mask *= valid_mask

                unknown_index = voxel_index[unknown_mask]

                ray_utility = self.cal_utility(
                    roi_index, unknown_index, len(voxel_index)
                )
                view_utility += ray_utility

            utility_list[i] = view_utility

        return utility_list

    def cal_utility(self, roi_index, unknown_index, total_voxel):
        dist = torch.cdist(
            1.0 * unknown_index.unsqueeze(0), 1.0 * roi_index.unsqueeze(0)
        ).squeeze(0)
        min_dist_to_roi, _ = torch.min(dist, dim=-1)
        score = 0.5 + torch.clamp(
            0.5 * (self.max_dist - min_dist_to_roi) / self.max_dist, min=0
        )
        ray_utility = torch.sum(score) / total_voxel
        return ray_utility.cpu().numpy()
