from ..planner_base import PlannerBase
from ..planning_utils import *


class CoveragePlanner(PlannerBase):
    def __init__(self, args, device):
        super(CoveragePlanner, self).__init__(args, device)
        self.random_rotation = np.random.uniform(0, 2 * np.pi)
        self.init_planning()
        self.get_view_list()

    def get_view_list(self):
        view_list = fibonacci_spiral_hemisphere(
            self.planning_budget, self.radius, self.phi_min
        )
        view_list[:, 1] += self.random_rotation
        view_list = view_list[:-1]
        view_list = np.flip(view_list, axis=0)
        print(view_list)
        self.view_list = iter(view_list)

    def plan_next_view(self):
        return [next(self.view_list)]
