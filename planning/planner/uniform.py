from ..planner_base import PlannerBase
from ..planning_utils import *


class UniformPlanner(PlannerBase):
    def __init__(self, args, device):
        super(UniformPlanner, self).__init__(args, device)
        self.num_candidates = args.num_candidates
        self.init_planning()

    def plan_next_view(self):
        view_list = np.empty((self.num_candidates, 2))
        for i in range(self.num_candidates):
            view_list[i] = uniform_sampling(self.radius, self.phi_min)

        nbv_index = np.random.choice(len(view_list))
        return [view_list[nbv_index]]
