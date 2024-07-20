from ..planner_base import PlannerBase
from ..planning_utils import *
from scipy.spatial import distance


class MaxDistancePlanner(PlannerBase):
    def __init__(self, args, device):
        super(MaxDistancePlanner, self).__init__(args, device)
        self.num_candidates = args.num_candidates
        self.init_planning()

    def get_camera_view_direction(self, poses):
        view_direction = poses[..., :3, 0]
        view_direction = view_direction / np.linalg.norm(view_direction)
        return view_direction

    def plan_next_view(self):
        view_list = np.empty((self.num_candidates, 2))

        for i in range(self.num_candidates):
            view_list[i] = uniform_sampling(self.radius, self.phi_min)

        pose_list = view_to_pose_batch(view_list, self.radius)
        # new_view_list = self.get_camera_view_direction(pose_list)

        current_pose_list = view_to_pose_batch(self.all_view_trajectory, self.radius)
        # current_view_list = get_camera_view_direction(current_pose_list)

        dist_list = []
        for pose in pose_list:
            diff = 0
            count = 0
            for ref_pose in current_pose_list:
                # print(view, ref_view)
                # cos_sim = 1 - distance.cosine(view, ref_view)
                view_diff = np.sum(np.square(pose[:3, 3] - ref_pose[:3, 3]))
                # view_diff = np.sum(np.abs(ref_view - view))
                # cos_dist = np.min((cos_dist, 1))
                diff += view_diff
                count += 1
            # print(dist)
            dist_list.append(diff / count)

        return [view_list[np.argmax(dist_list)]]
