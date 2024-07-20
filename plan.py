import torch
import numpy as np
import rospy

from tools.options import arg_parser
from tools.gui import GUI
from tools.exp_runner import ExperimentRunner
from planning import get_planner

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    args = arg_parser("planning")

    rospy.init_node(args.planner_type)
    planner = get_planner(args, device)
    if args.gui:
        gui = GUI(args, planner, planning_mode=True)
        gui.start()
    else:
        exp_runner = ExperimentRunner(planner)
        exp_runner.start()
