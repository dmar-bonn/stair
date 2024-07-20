import torch
import torchmetrics
import math
import json
import os
import numpy as np
from numpy import ma
from tqdm import tqdm
import imageio
import open3d as o3d
import cv2
import click
from tools.utils import *

# from dataLoader.ray_utils import *
from semantic_nerf import load_model

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
    "--batch_size",
    type=int,
    help="ray batch size for rendering",
    default=4096,
)
@click.option(
    "--visual_output",
    is_flag=True,
    help="save test rendereing outputs",
)
@click.option(
    "--use_patch",
    is_flag=True,
    help="use valid image patch for evaluation",
)
def main(exp_path, test_path, batch_size, visual_output, use_patch):
    with open(os.path.join(exp_path, f"transforms.json"), "r") as file:
        meta = json.load(file)
    target_class_id = meta["target_class_id"]
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

    for step_num in save_list:
        print(
            f"------------------------ evaluate rendering performance after planning step{step_num} ------------------------"
        )
        model_path = f"{exp_path}/models/model_{step_num}.th"
        model_file = torch.load(model_path)
        model = load_model(model_file, device)
        renderer = BatchRenderer

        with torch.no_grad():
            with open(os.path.join(test_path, f"transforms.json"), "r") as file:
                meta = json.load(file)
            w, h = meta["w"], meta["h"]
            focal_x = meta["fl_x"]
            focal_y = meta["fl_y"]
            c_x = meta["cx"]
            c_y = meta["cy"]
            near_far = [0.0, 5]

            directions = get_ray_directions(
                [h, w, [focal_x, focal_y], [c_x, c_y]]
            )  # (h, w, 3)

            if visual_output:
                visual_output_path = f"{exp_path}/visual_test/step{step_num}"
                os.makedirs(visual_output_path, exist_ok=True)

            # rendering metrics
            psnr_list = []
            ssim_list = []
            depth_error_list = []
            uncertainty_list = []
            for i in tqdm(range(len(meta["frames"])), desc="Evaluation"):
                frame = meta["frames"][i]

                pose = np.array(frame["transform_matrix"])
                c2w = torch.FloatTensor(pose)

                semantic_gt_path = os.path.join(
                    test_path, f"{frame['gt_semantic_path']}"
                )
                semantic_gt = np.load(semantic_gt_path)
                target_mask_gt = np.isin(semantic_gt, target_class_id)

                depth_gt_path = os.path.join(test_path, f"{frame['depth_path']}")
                depth_gt = np.load(depth_gt_path)
                depth_gt[np.isinf(depth_gt)] = 0

                rgb_gt_path = os.path.join(test_path, f"{frame['image_path']}")
                rgb_gt = imageio.v2.imread(rgb_gt_path)[..., :3]
                rgb_gt = rgb_gt / 255
                rgb_gt[~target_mask_gt, :] = np.array(
                    [1.0, 1.0, 1.0]
                )  # white background

                # rendering output from trained nerf
                rays_o, rays_d = get_rays(directions, c2w)  # both (h*w, 3)
                all_rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)
                render_results = renderer(
                    all_rays,
                    model,
                    require_class=target_class_id,
                    chunk=batch_size,
                    device=device,
                )
                rgb_pred = render_results[0].view(h, w, 3).cpu().numpy()
                uncertainty_pred = render_results[3].view(h, w).cpu().numpy()
                semantic_pred = torch.argmax(render_results[1].cpu(), dim=-1).view(h, w)

                depth_pred = render_results[2].view(h, w).cpu().numpy()

                depth_ae_map = np.abs(depth_gt - depth_pred)
                color_ae_map = np.mean(np.abs(rgb_gt - rgb_pred), axis=-1)

                if visual_output:
                    rgb_gt_record = (rgb_gt * 255).astype("uint8")
                    imageio.imwrite(
                        f"{visual_output_path}/{i+1:04d}_rgb_gt.png", rgb_gt_record
                    )
                    uncertainty_pred_map = visualize_depth_unc_numpy(uncertainty_pred)
                    uncertainty_pred_map = (uncertainty_pred_map * 255).astype("uint8")
                    imageio.imwrite(
                        f"{visual_output_path}/{i+1:04d}_uncertainty_pred.png",
                        uncertainty_pred_map,
                    )

                    depth_error_map = visualize_error_numpy(depth_ae_map, near_far)
                    depth_error_map = (depth_error_map * 255).astype("uint8")
                    imageio.imwrite(
                        f"{visual_output_path}/{i+1:04d}_depth_error.png",
                        depth_error_map,
                    )
                    rgb_error_map = visualize_error_numpy(color_ae_map)
                    rgb_error_map = (rgb_error_map * 255).astype("uint8")
                    imageio.imwrite(
                        f"{visual_output_path}/{i+1:04d}_rgb_error.png", rgb_error_map
                    )

                    rgb_pred_record = (rgb_pred * 255).astype("uint8")
                    imageio.imwrite(
                        f"{visual_output_path}/{i+1:04d}_rgb_pred.png", rgb_pred_record
                    )

                    depth_pred_record = (
                        visualize_depth_numpy(depth_pred, near_far) * 255
                    ).astype("uint8")
                    imageio.imwrite(
                        f"{visual_output_path}/{i+1:04d}_depth_pred.png",
                        depth_pred_record,
                    )

                psnr, ssim = cal_rendering_metrics(
                    torch.tensor(rgb_pred), torch.tensor(rgb_gt)
                )
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                depth_error_list.append(np.mean(depth_ae_map))

                uncertainty_list.append(np.mean(uncertainty_pred))

            avg_psnr = np.mean(psnr_list)
            avg_ssim = np.mean(ssim_list)
            avg_uncertainty = np.mean(uncertainty_list, dtype=np.float64)
            avg_depth_error = np.mean(depth_error_list, dtype=np.float64)
            rendering_results = {
                "average_psnr": avg_psnr,
                "average_ssim": avg_ssim,
                "average_uncertainty": avg_uncertainty,
                "average_depth_error": avg_depth_error,
            }
            print(rendering_results)

            results_all[str(step_num)].update(rendering_results)

    with open(f"{exp_path}/results.json", "w") as file:
        json.dump(results_all, file, indent=4)


def cal_rendering_metrics(rgb_pred, rgb_gt):
    mse = ((rgb_pred - rgb_gt) ** 2).mean()
    psnr = -10 * math.log10(mse)
    ssim_cal = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0)
    ssim = ssim_cal(
        rgb_pred.permute(2, 0, 1).unsqueeze(0), rgb_gt.permute(2, 0, 1).unsqueeze(0)
    ).item()
    return psnr, ssim


def get_valid_image_patch(img):
    segmentation = np.where(img.numpy() != np.array([1.0, 1.0, 1.0]))
    if (
        len(segmentation) != 0
        and len(segmentation[1]) != 0
        and len(segmentation[0]) != 0
    ):
        x_min = int(np.min(segmentation[1]))
        x_max = int(np.max(segmentation[1]))
        y_min = int(np.min(segmentation[0]))
        y_max = int(np.max(segmentation[0]))

        return x_min, x_max, y_min, y_max

    else:
        return 0, -1, 0, -1


if __name__ == "__main__":
    main()
