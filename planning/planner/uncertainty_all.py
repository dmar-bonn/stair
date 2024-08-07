from scipy.spatial import distance
from ..planner_base import PlannerBase
from ..planning_utils import *


class UncertaintyAllPlanner(PlannerBase):
    def __init__(self, args, device):
        super(UncertaintyAllPlanner, self).__init__(args, device)
        self.num_candidates = args.num_candidates
        self.downscale_planning = args.downscale_planning
        self.rendering_intrinsics = np.array([*self.focal, self.W // 2, self.H // 2])
        self.importance_sampling = args.importance_sampling
        self.candidate_ratio = args.candidate_ratio
        self.cluster_top_k = args.cluster_top_k
        self.init_planning()

    def plan_next_view(self):
        if self.importance_sampling:
            num_candidate_uniform = np.floor(
                self.num_candidates * self.candidate_ratio
            ).astype(int)
            num_candidate_importance = np.floor(
                (self.num_candidates - num_candidate_uniform) / self.cluster_top_k,
            ).astype(int)
        else:
            num_candidate_uniform = self.num_candidates
            num_candidate_importance = 0

        view_list = np.empty((num_candidate_uniform, 2))

        for i in range(num_candidate_uniform):
            view_list[i] = uniform_sampling(self.radius, self.phi_min)

        candidate_poses = view_to_pose_batch(view_list, self.radius)
        candidate_poses = gazebo2opencv(candidate_poses).astype(np.float32)
        utility_list = self.get_candidate_utility(candidate_poses)

        if self.importance_sampling:
            nbv_indeces = np.argsort(utility_list)[::-1]
            top_k_indeces = nbv_indeces[: self.cluster_top_k]

            view_list = view_list[top_k_indeces]
            candidate_poses = candidate_poses[top_k_indeces]
            utility_list = utility_list[top_k_indeces]

            local_view_list = np.empty(
                (num_candidate_importance * self.cluster_top_k, 2)
            )
            for k, pose in enumerate(candidate_poses):
                for i in range(num_candidate_importance):
                    local_view = local_uniform_sampling(
                        self.radius, pose[:3, 3], [0.1, 0.5], self.phi_min
                    )
                    local_view_list[k * num_candidate_importance + i] = local_view

            local_candidate_poses = view_to_pose_batch(local_view_list, self.radius)
            local_candidate_poses = gazebo2opencv(local_candidate_poses).astype(
                np.float32
            )
            local_utility_list = self.get_candidate_utility(local_candidate_poses)

            utility_list = np.concatenate((utility_list, local_utility_list), axis=0)
            view_list = np.concatenate((view_list, local_view_list), axis=0)

        return self.select_nbv(utility_list, view_list)

    def select_nbv(self, utitlity_list, view_list):
        nbv_indeces = np.argsort(utitlity_list)[::-1]

        test_pose_list = view_to_pose_batch(view_list, self.radius)
        test_view_list = get_camera_view_direction(test_pose_list)

        current_pose_list = view_to_pose_batch(self.all_view_trajectory, self.radius)
        current_view_list = get_camera_view_direction(current_pose_list)

        for index in nbv_indeces:
            test_view = test_view_list[index]

            # for current_view in current_view_list:
            cos_dist_list = np.array(
                [
                    distance.cosine(test_view, current_view)
                    for current_view in current_view_list
                ]
            )
            if np.all(cos_dist_list > 0.01):
                return [view_list[index]]
            # else:
            #     print("reject!!!!")

        self.plan_next_view()

    def get_candidate_utility(self, candidate_poses):
        rendering_results = self.mapper.render_view(
            candidate_poses,
            self.rendering_intrinsics,
            self.H,
            self.W,
            self.downscale_planning,
            -1,
            planning=True,
        )
        depth_unc = rendering_results["depth_uncertainty"]
        semantic_unc = rendering_results["semantic_uncertainty"]
        semantic = rendering_results["semantic"]
        object_ratio = rendering_results["object_ratio"]
        utility_list = self.cal_utility(
            len(candidate_poses), depth_unc, semantic_unc, semantic, object_ratio
        )
        return utility_list

    def cal_utility(self, num, depth_unc, semantic_unc, semantic, object_ratio):
        depth_unc = depth_unc.reshape(num, -1)
        semantic_unc = semantic_unc.reshape(num, -1)
        depth_utility = np.sum(depth_unc, axis=-1)
        utility_list = depth_utility

        return utility_list
