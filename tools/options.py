import argparse
from configs import *
from dotmap import DotMap


def arg_parser(mode=None):
    parser = argparse.ArgumentParser()

    # parse commnad line arguments
    if mode is not None:
        parser.add_argument(
            "--config", type=str, nargs="+", required=True, help="config file name"
        )
    parser.add_argument(
        "--random_seed", type=int, default=20211202, help="random seed number"
    )
    parser.add_argument("--exp_name", default="test", type=str, help="experiment name")
    parser.add_argument("--exp_id", default=None, type=int, help="experiment id")
    parser.add_argument("--batch_size", default=8000, type=int, help="ray batch size")

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="specific weights npy file to reload for coarse network",
    )
    parser.add_argument(
        "--test_folder",
        type=str,
        default=None,
        help="test data folder",
    )
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--visualize_planning_results", action="store_true")
    parser.add_argument("--save_scene", action="store_true")
    parser.add_argument(
        "--planning_budget", "-BG", type=int, default=10, help="planning budget"
    )
    parser.add_argument("--target_class_id", nargs="+", type=int, default=[4])

    args = parser.parse_args()

    # parse configuration file arguments
    if mode is not None:
        cfg = get_cfg(mode, args.config)
        cfg.update(vars(args))
        args = DotMap(cfg, _dynamic=False)

    return args
