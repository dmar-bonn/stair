import yaml
import os

cfg_dir = os.path.dirname(__file__)
basic_cfg_path = os.path.join(cfg_dir, "basic.yaml")

with open(basic_cfg_path, "r") as file:
    basic_cfg = yaml.safe_load(file)


def get_cfg(mode, cfg_name_list):
    if mode == "training":
        assert len(cfg_name_list) == 1
        cfg = get_training_cfg(cfg_name_list[0])
    elif mode == "planning":
        assert len(cfg_name_list) == 2
        cfg = get_planning_cfg(cfg_name_list)
    else:
        RuntimeError("mode not support!")
    return cfg


def get_training_cfg(cfg_name):
    cfg_all = {}
    cfg_all.update(basic_cfg)

    cfg_path = os.path.join(cfg_dir, f"training/{cfg_name}.yaml")
    with open(cfg_path, "r") as file:
        cfg = yaml.safe_load(file)

    cfg_all.update(cfg)
    return cfg_all


def get_planning_cfg(cfg_names):
    planner_type, mapper_type = cfg_names
    print(planner_type, mapper_type)
    cfg_all = {}
    cfg_all.update(basic_cfg)

    planner_cfg_path = os.path.join(cfg_dir, f"planning/{planner_type}.yaml")
    with open(planner_cfg_path, "r") as file:
        planner_cfg = yaml.safe_load(file)
    cfg_all.update(planner_cfg)

    mapper_cfg_path = os.path.join(cfg_dir, f"mapping/{mapper_type}.yaml")
    with open(mapper_cfg_path, "r") as file:
        mapper_cfg = yaml.safe_load(file)
    cfg_all.update(mapper_cfg)
    return cfg_all
