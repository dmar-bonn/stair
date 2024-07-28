from .planner.uniform import UniformPlanner
from .planner.coverage import CoveragePlanner
from .planner.max_distance import MaxDistancePlanner
from .planner.uncertainty_all import UncertaintyAllPlanner
from .planner.uncertainty_target import UncertaintyTargetPlanner
from .planner.roi import ROIPlanner


def get_planner(args, device):
    planner_type = args.planner_type

    if planner_type == "uniform":
        return UniformPlanner(args, device)
    elif planner_type == "coverage":
        return CoveragePlanner(args, device)
    elif planner_type == "max_distance":
        return MaxDistancePlanner(args, device)
    elif planner_type == "uncertainty_target":
        return UncertaintyTargetPlanner(args, device)
    elif planner_type == "uncertainty_all":
        return UncertaintyAllPlanner(args, device)
    elif planner_type == "roi":
        return ROIPlanner(args, device)
    else:
        RuntimeError("planner type not defined")
