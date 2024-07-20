import torch
import json
import os
import numpy as np
import open3d as o3d
import click


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@click.command()
@click.option("--exp_path", required=True, type=str, help="path to pretrained nerf")
@click.option(
    "--test_path",
    required=True,
    type=str,
    help="path to test folder with test views information",
)
@click.option(
    "--threshold",
    type=float,
    help="threshold for calculating mesh quality",
    default=0.01,
)
def main(exp_path, test_path, threshold):
    with open(os.path.join(exp_path, f"transforms.json"), "r") as file:
        meta = json.load(file)
    save_list = meta["save_list"]

    results_all = {}
    for step_num in save_list:
        results_all[str(step_num)] = {}

    file_name = f"{exp_path}/results.json"
    try:
        with open(file_name, "r") as f:
            results_all = json.load(f)
    except IOError:
        pass

    mesh_gt = o3d.io.read_triangle_mesh(f"{test_path}/mesh.ply")
    pcd_gt = mesh_to_pcd(mesh_gt)

    for step_num in save_list:
        print(
            f"------------------------ evaluate mesh quality after planning step{step_num} ------------------------"
        )
        mesh_path = f"{exp_path}/meshes/mesh_{step_num}.ply"
        mesh_pred = o3d.io.read_triangle_mesh(mesh_path)

        if mesh_pred.is_empty():
            chamfer_distance = 1
            precision = 0
            recall = 0
            f1 = 0
        else:
            pcd_pred = mesh_to_pcd(mesh_pred)

            chamfer_distance, precision, recall, f1 = cal_geometry_metrics(
                pcd_pred, pcd_gt, threshold
            )

        mesh_results = {
            "chamfer_distance": chamfer_distance,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        print(mesh_results)

        results_all[str(step_num)].update(mesh_results)

    with open(f"{exp_path}/results.json", "w") as file:
        json.dump(results_all, file, indent=4)


def cal_geometry_metrics(pcd_pred, pcd_gt, threshold):
    dist_pred_2_gt = np.asarray(pcd_pred.compute_point_cloud_distance(pcd_gt))
    dist_gt_2_pred = np.asarray(pcd_gt.compute_point_cloud_distance(pcd_pred))
    chamfer_distance = (np.mean(dist_gt_2_pred) + np.mean(dist_pred_2_gt)) / 2

    p = np.where(dist_pred_2_gt < threshold)[0]
    precision = 100 / len(dist_pred_2_gt) * len(p)

    r = np.where(dist_gt_2_pred < threshold)[0]
    recall = 100 / len(dist_gt_2_pred) * len(r)

    if precision == 0 or recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return chamfer_distance, precision, recall, f1


def mesh_to_pcd(mesh):
    pcd = mesh.sample_points_uniformly(1000000)
    return pcd


if __name__ == "__main__":
    main()
