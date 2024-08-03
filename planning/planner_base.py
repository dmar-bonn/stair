import time
import os
import datetime
import numpy as np
import imageio
import yaml
import warnings

from mapping import get_mapper

from tools.utils import *
from .simulator_bridge import SimulatorBridge
from .planning_utils import *

warnings.filterwarnings("ignore")


class PlannerBase:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.verbose = args.verbose
        self.record_path = os.path.expanduser(
            os.path.join(
                args.exp_path,
                f"{args.planner_type}",
                (
                    str(args.exp_id)
                    if args.exp_id is not None
                    else datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                ),
            )
        )
        self.visualize_measurement = args.visualize_measurement
        self.H_gui = args.H_gui
        self.W_gui = args.W_gui
        self.save_scene = args.save_scene

        # planning setup
        self.initial_type = args.initial_type
        self.map_update_for_planning = args.map_update_for_planning
        self.iteration_per_step = args.iteration_per_step
        self.planning_target_id = args.target_class_id
        self.class_num = args.class_num
        self.use_pseudo_label = args.use_pseudo_label
        self.save_list = args.save_list
        self.planning_budget = self.save_list[-1]
        self.online_experiment = args.online_experiment

        # components
        self.mapper = get_mapper(args, device)
        self.simulator_bridge = SimulatorBridge(args)

        # sensor setup
        self.camera_info = self.simulator_bridge.camera_info
        print(f"camera info: {self.camera_info}")
        self.H, self.W = self.camera_info["image_resolution"]
        self.focal = self.camera_info["focal"]
        self.c = self.camera_info["c"]
        self.camera_intrinsics = (self.H, self.W, self.focal, self.c)
        self.ray_direction = get_ray_directions(self.camera_intrinsics)
        self.pixel_num = self.H * self.W

        # hemisphere action space
        self.min_height = args.min_height
        self.radius = args.radius
        self.phi_min = np.arcsin(self.min_height / self.radius)
        self.phi_max = 0.5 * np.pi
        self.theta_min = 0
        self.theta_max = 2 * np.pi

        self.fine_stage = False  # offline traininig

    def init_planning(self):
        print("---------- initialize planner ----------")
        self.trajecory_viewer = TrajectoryVisualizer(self.H_gui, self.W_gui)

        # for data recording
        self.trajectory = np.empty((self.planning_budget, 4, 4))
        self.view_trajectory = np.empty((self.planning_budget, 2))  # [phi, theta]
        self.rgb_measurements = np.empty((self.planning_budget, self.H, self.W, 3))
        self.depth_measurements = np.empty((self.planning_budget, self.H, self.W))
        self.semantic_measurements = np.empty((self.planning_budget, self.H, self.W))
        self.semantic_gt = np.empty((self.planning_budget, self.H, self.W))
        self.semantic_mask = np.ones((self.planning_budget, self.H, self.W))
        self.semantic_unc_measurements = np.zeros(
            (self.planning_budget, self.H, self.W)
        )

        # for nerf training
        self.all_rays_record = torch.empty((self.planning_budget * self.pixel_num, 6))
        self.all_rgbs_record = torch.empty((self.planning_budget * self.pixel_num, 3))
        self.all_depths_record = torch.empty((self.planning_budget * self.pixel_num))
        self.all_semantics_gt_record = torch.empty(
            (self.planning_budget * self.pixel_num)
        )  # label index
        self.all_semantics_record = torch.empty(
            (self.planning_budget * self.pixel_num, self.class_num)
        )  # probability vector
        self.all_semantics_mask_record = torch.ones(
            (self.planning_budget * self.pixel_num)
        )
        self.all_semantics_unc_record = torch.zeros(
            (self.planning_budget * self.pixel_num)
        )
        self.step = 0
        self.update_iteration = 0
        self.initialization = True

    def planning(self):
        if self.initialization:
            steps = self.init_camera_pose()
            self.initialization = False
            return steps
        else:
            next_views = self.plan_next_view()
            for next_view in next_views:
                self.move_sensor(next_view)
            return len(next_views)

    def init_camera_pose(self):
        print("------ start mission ------ \n")
        print("------ initialize camera pose ------ \n")

        initial_view = [0.5 * np.pi, 0]
        self.move_sensor(initial_view)
        return 1

    def move_sensor(self, view):
        pose = view_to_pose(view, self.radius)
        self.simulator_bridge.move_camera(pose)

        self.current_view = view
        self.current_pose = pose
        if self.verbose:
            print(pose)
            print(
                f"------ reach given pose and take measurement No.{self.step + 1} ------\n"
            )
        time.sleep(2)  # lazy solution to make sure we receive correct images from ROS
        rgb, depth, semantic = self.simulator_bridge.get_measurement()
        self.record_step(view, pose, rgb, depth, semantic)
        self.update_measurement()
        self.step += 1

    def plan_next_view(self):
        raise NotImplementedError("plan_next_view method is not implemented")

    def update_measurement(self):
        buffer = (
            self.all_rays,
            self.all_rgbs,
            self.all_semantics,
            self.all_semantics_unc,
            self.all_depths,
            self.all_semantics_mask,
            self.all_semantics_gt,
            self.all_trajectory,
            self.camera_intrinsics,
        )
        self.mapper.update_measurement_buffer(buffer)

    def record_step(self, view, pose, rgb, depth, semantic):
        self.record_trajectory(view, pose)
        self.record_rgb_measurement(rgb)
        self.record_depth_measurement(depth)
        self.record_semantic_gt(semantic)

        if self.use_pseudo_label:
            print("---------- segment RGB measurement ----------")
            pseudo_label_prob, uncertainty = self.segmenter.predict(rgb)
            pseudo_label_prob = pseudo_label_prob.squeeze(0).permute(
                1, 2, 0
            )  # (H, W, C)
            uncertainty = uncertainty.squeeze(0)

            self.record_semantic_measurement(pseudo_label_prob.numpy())
            self.record_semantic_unc_measurement(uncertainty.numpy())
            self.record_semantic_mask_measurement(
                pseudo_label_prob, torch.tensor(semantic), uncertainty
            )
        else:
            semantic_prob = label2onehot(semantic, self.class_num)
            self.record_semantic_measurement(semantic_prob)

    def record_rgb_measurement(self, rgb):
        rgb = np.clip(rgb, a_min=0, a_max=255)
        rgb = rgb / 255
        self.rgb_measurements[self.step] = rgb
        self.all_rgbs_record[
            self.step * self.pixel_num : (self.step + 1) * self.pixel_num
        ] = torch.tensor(rgb.reshape(-1, 3))

    def record_depth_measurement(self, depth):
        depth[np.isinf(depth)] = 0
        depth[np.isnan(depth)] = 0
        self.depth_measurements[self.step] = depth
        self.all_depths_record[
            self.step * self.pixel_num : (self.step + 1) * self.pixel_num
        ] = torch.tensor(depth).flatten()

    def record_semantic_mask_measurement(self, pseudo, gt, unc):
        _semantic_label = torch.argmax(pseudo, dim=-1)
        valid_semantic_mask = torch.eq(_semantic_label, gt)

        valid_semantic_mask = valid_semantic_mask.long()

        self.semantic_mask[self.step] = valid_semantic_mask.numpy()

        self.all_semantics_mask_record[
            self.step * self.pixel_num : (self.step + 1) * self.pixel_num
        ] = valid_semantic_mask.flatten()

    def record_semantic_gt(self, semantic):
        self.semantic_gt[self.step] = semantic
        self.all_semantics_gt_record[
            self.step * self.pixel_num : (self.step + 1) * self.pixel_num
        ] = torch.tensor(semantic).flatten()

    def record_semantic_measurement(self, semantic):
        self.semantic_measurements[self.step] = np.argmax(semantic, axis=-1)
        self.all_semantics_record[
            self.step * self.pixel_num : (self.step + 1) * self.pixel_num
        ] = torch.tensor(semantic).reshape(-1, self.class_num)

    def record_semantic_unc_measurement(self, semantic_unc):
        self.semantic_unc_measurements[self.step] = semantic_unc

        if True:
            semantic_unc = F.avg_pool2d(
                torch.tensor(semantic_unc)[None, None, ...],
                kernel_size=3,
                padding=3 // 2,
                stride=1,
            ).view(self.H, self.W)

        self.semantic_unc_measurements[self.step] = semantic_unc.numpy()

        self.all_semantics_unc_record[
            self.step * self.pixel_num : (self.step + 1) * self.pixel_num
        ] = torch.tensor(semantic_unc).flatten()

    def record_trajectory(self, view, pose):
        pose = gazebo2opencv(pose)
        self.trajectory[self.step] = pose
        self.trajecory_viewer.add_view(pose)
        self.view_trajectory[self.step] = view

        pose_tensor = torch.tensor(pose, dtype=torch.float32)
        rays_o, rays_d = get_rays(
            self.ray_direction,
            pose_tensor,
        )  # both (h*w, 3)
        self.all_rays_record[
            self.step * self.pixel_num : (self.step + 1) * self.pixel_num
        ] = torch.cat([rays_o, rays_d], dim=-1)

    @property
    def max_iteration(self):
        return self.step * self.iteration_per_step

    def update(self, train_steps):
        outputs = None
        if self.update_iteration < self.max_iteration:
            outputs, iteration = self.mapper.update_map(train_steps)
            self.update_iteration = iteration
        return outputs, self.update_iteration

    def update_end(self):
        if self.online_experiment and self.step in self.save_list:
            self.save_model(self.step)
            self.save_mesh(self.step)

    def render(self, pose, intrinsic, H, W, downscale, n_samples, rendering_target_id):
        outputs = self.mapper.render_view(
            pose, intrinsic, H, W, downscale, n_samples, rendering_target_id
        )

        trajectory_map = self.trajecory_viewer.get_rendering(pose)
        if self.visualize_measurement:
            trajectory_map = overlap_images(
                trajectory_map,
                [
                    self.rgb_measurements[self.step - 1],
                    label2color(
                        self.semantic_gt[self.step - 1],
                        "shapenet",
                    ),
                    visualize_depth_numpy(
                        self.depth_measurements[self.step - 1], [0.0, 4.0]
                    ),
                ],
            )

        outputs["trajectory"] = trajectory_map

        return outputs

    @property
    def all_view_trajectory(self):
        return self.view_trajectory[: self.step + 1]

    @property
    def all_trajectory(self):
        return self.trajectory[: self.step + 1]

    @property
    def all_rays(self):
        return self.all_rays_record[: (self.step + 1) * self.pixel_num]

    @property
    def all_rgbs(self):
        return self.all_rgbs_record[: (self.step + 1) * self.pixel_num]

    @property
    def all_semantics_mask(self):
        return self.all_semantics_mask_record[: (self.step + 1) * self.pixel_num]

    @property
    def all_semantics_gt(self):
        return self.all_semantics_gt_record[: (self.step + 1) * self.pixel_num]

    @property
    def all_semantics(self):
        return self.all_semantics_record[: (self.step + 1) * self.pixel_num]

    @property
    def all_semantics_unc(self):
        return self.all_semantics_unc_record[: (self.step + 1) * self.pixel_num]

    @property
    def all_depths(self):
        return self.all_depths_record[: (self.step + 1) * self.pixel_num]

    def reset(self):
        self.update_iteration = 0
        self.mapper.init_traning_setup(fine_stage=self.fine_stage)

    def save_model(self, model_name="final"):
        print("------ save model ------\n")
        model_folder = f"{self.record_path}/models"
        os.makedirs(model_folder, exist_ok=True)
        model_file = f"{model_folder}/model_{model_name}.th"

        self.mapper.save_model(model_file)

    def save_mesh(self, mesh_name="final"):
        print("------ save mesh ------\n")
        mesh_folder = f"{self.record_path}/meshes"
        os.makedirs(mesh_folder, exist_ok=True)
        mesh_file = f"{mesh_folder}/mesh_{mesh_name}.ply"

        if self.save_scene:
            target_id = []
        else:
            target_id = self.planning_target_id

        self.mapper.save_mesh(mesh_file, target_id)

    def save_experiment(self):
        print("------ record experiment data ------\n")

        os.makedirs(self.record_path, exist_ok=True)

        # save data record during planning
        images_path = os.path.join(self.record_path, "images")
        os.makedirs(images_path, exist_ok=True)
        semantics_path = os.path.join(self.record_path, "semantics")
        os.makedirs(semantics_path, exist_ok=True)
        depths_path = os.path.join(self.record_path, "depths")
        os.makedirs(depths_path, exist_ok=True)

        for i, rgb in enumerate(self.rgb_measurements[: self.step]):
            rgb = np.round(rgb * 255).astype("uint8")
            imageio.imwrite(f"{images_path}/{i+1:04d}.png", rgb)

        for i, depth in enumerate(self.depth_measurements[: self.step]):
            with open(f"{depths_path}/depth_{i+1:04d}.npy", "wb") as f:
                depth_array = np.array(depth, dtype=np.float32)
                np.save(f, depth_array)

        for i, semantic in enumerate(self.semantic_measurements[: self.step]):
            with open(f"{semantics_path}/pseudo_{i+1:04d}.npy", "wb") as f:
                semantic_array = np.array(semantic, dtype=np.int8)
                np.save(f, semantic_array)

            # for visualizing semantic label
            semantic_map = (label2color(semantic) * 255).astype("uint8")
            imageio.imwrite(f"{semantics_path}/pseudo_{i+1:04d}.png", semantic_map)

        for i, semantic_gt in enumerate(self.semantic_gt[: self.step]):
            with open(f"{semantics_path}/gt_{i+1:04d}.npy", "wb") as f:
                semantic_gt_array = np.array(semantic_gt, dtype=np.int8)
                np.save(f, semantic_gt_array)

            semantic_gt_map = (label2color(semantic_gt) * 255).astype("uint8")
            imageio.imwrite(f"{semantics_path}/gt_{i+1:04d}.png", semantic_gt_map)

        with open(f"{self.record_path}/trajectory.npy", "wb") as f:
            np.save(f, self.trajectory)

        with open(f"{self.record_path}/camera_info.yaml", "w") as f:
            yaml.safe_dump(self.camera_info, f)

        # record json data required for off-line training
        record_meta_data(
            self.record_path,
            self.camera_info,
            self.trajectory[: self.step + 1],
            self.planning_target_id,
            self.save_list,
        )

        # record experiment configuration
        with open(f"{self.record_path}/configuration.yaml", "w") as f:
            yaml.safe_dump(self.args.toDict(), f)
