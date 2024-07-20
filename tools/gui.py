import os
import math
import torch
import numpy as np
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R
import time
from PIL import Image

from .utils import LABELS


########## part of code is based on torch-ngp(https://github.com/ashawkey/torch-ngp) ##########


class OrbitCamera:
    def __init__(self, H, W, r=2, fovx=60, fovy=60):
        self.H = H
        self.W = W
        self.fovx = fovx  # in degree
        self.fovy = fovy  # in degree
        self.r = r
        self.up = np.array([0, 1, 0], dtype=np.float32)
        self.to_initial_pose()

    def to_initial_pose(self):
        self.radius = self.r  # camera distance from center
        self.center = np.array([0, 0, 0], dtype=np.float32)
        self.rot = R.from_quat(
            [1, 0, 0, 0]
        )  # init camera matrix: [[1, 0, 0], [0, -1, 0], [0, 0, 1]] (to suit ngp convention)

    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    @property
    def intrinsics(self):
        focal_x = self.W / (2 * np.tan(np.radians(self.fovx) / 2))
        focal_y = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal_x, focal_y, self.W // 2, self.H // 2])

    # define camera movement
    def orbit(self, dx, dy):
        side = self.rot.as_matrix()[:3, 0]
        rotvec_x = self.up * np.radians(-0.1 * dx)
        rotvec_y = side * np.radians(-0.1 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, dy, dz])


class GUI:
    def __init__(self, args, trainer, planning_mode=False):
        # visualization setup
        self.H = args.H_gui
        self.W = args.W_gui
        self.downscale = 1
        self.max_spp = args.max_spp
        self.render_buffer_main = np.zeros((self.H, self.W, 3), dtype=np.float32)
        self.render_buffer_depth_unc = np.zeros((self.H, self.W, 3), dtype=np.float32)
        self.render_buffer_semantic = np.zeros((self.H, self.W, 3), dtype=np.float32)
        self.render_buffer_semantic_unc = np.zeros(
            (self.H, self.W, 3), dtype=np.float32
        )
        self.render_buffer_trajectory = np.zeros((self.H, self.W, 3), dtype=np.float32)
        self.mode = "rgb"  # choose from ['rgb', 'depth', "acc"]
        self.dynamic_resolution = True
        self.need_update = True
        self.cam = OrbitCamera(
            H=args.H_gui,
            W=args.W_gui,
            r=args.radius_gui,
            fovx=args.fovx_gui,
            fovy=args.fovy_gui,
        )

        # planning setup
        self.trainer = trainer
        self.require_training = False
        self.train_steps = 10
        self.step = 0  # training step
        self.planning_target_id = []
        self.rendering_target_id = []
        self.rendering_target = "all"
        self.training_done = True
        self.render_done = True
        self.planning_mode = planning_mode
        if self.planning_mode:
            self.require_planning = False
            self.planning_step = 0
            self.start_mission = False
            self.planning_budget = self.trainer.planning_budget

        # spawn gui
        dpg.create_context()
        self.register_dpg()
        self.render_step()

    def __del__(self):
        dpg.destroy_context()

    def register_dpg(self):
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.render_buffer_main,
                format=dpg.mvFormat_Float_rgb,
                tag="_main_texture",
            )

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.render_buffer_depth_unc,
                format=dpg.mvFormat_Float_rgb,
                tag="_depth_unc_texture",
            )

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.render_buffer_semantic,
                format=dpg.mvFormat_Float_rgb,
                tag="_semantic_texture",
            )

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.render_buffer_trajectory,
                format=dpg.mvFormat_Float_rgb,
                tag="_trajectory_texture",
            )

        with dpg.window(
            label="RGB / Depth / Acc.", tag="_main_window", width=self.W, height=self.H
        ):
            dpg.add_image("_main_texture")
        dpg.set_item_pos("_main_window", [self.W, 0])

        with dpg.window(
            label="Depth Uncertainty",
            tag="_depth_unc_window",
            width=self.W,
            height=self.H,
        ):
            dpg.add_image("_depth_unc_texture")
        dpg.set_item_pos("_depth_unc_window", [self.W, self.H])

        with dpg.window(
            label="Semantic", tag="_semantic_window", width=self.W, height=self.H
        ):
            dpg.add_image("_semantic_texture")
        dpg.set_item_pos("_semantic_window", [2 * self.W, 0])

        with dpg.window(
            label="Camera Trajectory",
            tag="_trajectory_window",
            width=self.W,
            height=self.H,
        ):
            dpg.add_image("_trajectory_texture")
        dpg.set_item_pos("_trajectory_window", [2 * self.W, self.H])

        with dpg.window(
            label="Control", tag="_control_window", width=self.W, height=self.H
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # time
            with dpg.group(horizontal=True):
                dpg.add_text("Train time: ")
                dpg.add_text("no data", tag="_log_train_time")

            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            with dpg.collapsing_header(label="Taget", default_open=True):
                with dpg.group(horizontal=True):

                    def callback_select_class(sender, app_data):
                        target_class_id = LABELS["shapenet"][app_data]["id"]
                        self.planning_target_id.append(target_class_id)
                        print("planning target id:", self.planning_target_id)
                        self.trainer.planning_target_id = self.planning_target_id
                        self.need_update = True

                    dpg.add_combo(
                        ("car", "table", "chair", "camera", "airplane", "sofa"),
                        label="target class",
                        default_value=None,
                        callback=callback_select_class,
                    )

            if self.planning_mode:
                with dpg.collapsing_header(label="View Planning", default_open=True):
                    with dpg.group(horizontal=True):

                        def callback_view_budget(sender, app_data, user_data):
                            budget = dpg.get_value(user_data[0])
                            self.planning_budget = budget
                            self.trainer.planning_budget = budget
                            self.trainer.init_planning()
                            dpg.set_value("_log_set", f"view budget set to {budget}")
                            time.sleep(1)
                            dpg.set_value("_log_set", "")

                        budget = dpg.add_input_int(
                            label="Planning Budget",
                            width=100,
                            default_value=self.planning_budget,
                        )

                        dpg.add_button(
                            label="set",
                            tag="_button_set",
                            callback=callback_view_budget,
                            user_data=[budget],
                        )
                        dpg.bind_item_theme("_button_set", theme_button)
                        dpg.add_text("", tag="_log_set")

                    with dpg.group(horizontal=True):
                        dpg.add_text("Mission: ")

                        def callback_start_mission(sender, app_data):
                            self.start_mission = True

                        dpg.add_button(
                            label="start",
                            tag="_button_start_mission",
                            callback=callback_start_mission,
                        )
                        dpg.bind_item_theme("_button_start_mission", theme_button)

                    with dpg.group(horizontal=True):
                        dpg.add_text("Single-Step Plan: ")

                        def callback_plan(sender, app_data):
                            self.require_planning = True
                            dpg.configure_item("_button_train", label="stop")

                        dpg.add_button(
                            label="next step",
                            tag="_button_plan",
                            callback=callback_plan,
                        )
                        dpg.bind_item_theme("_button_plan", theme_button)

                        dpg.add_text(
                            f"planning step: {self.planning_step}", tag="_log_plan_step"
                        )

                    with dpg.group(horizontal=True):
                        dpg.add_text("Experiment: ")

                        def callback_save_exp(sender, app_data):
                            self.trainer.save_experiment()
                            dpg.set_value("_log_exp", "saved")
                            time.sleep(1)
                            dpg.set_value("_log_exp", "")

                        dpg.add_button(
                            label="save",
                            tag="_button_save_exp",
                            callback=callback_save_exp,
                        )
                        dpg.bind_item_theme("_button_save_exp", theme_button)

                        dpg.add_text("", tag="_log_exp")

                        def callback_reset_exp(sender, app_data):
                            self.trainer.init_planning()
                            self.trainer.reset()
                            self.step = 0
                            self.planning_step = 0
                            self.require_training = False
                            self.require_planning = False
                            self.start_mission = False
                            self.need_update = True
                            dpg.set_value(
                                "_log_plan_step",
                                f"planning step: {self.planning_step}",
                            )
                            dpg.configure_item("_button_train", label="start")

                        dpg.add_button(
                            label="reset",
                            tag="_button_reset_exp",
                            callback=callback_reset_exp,
                        )
                        dpg.bind_item_theme("_button_reset_exp", theme_button)

            # train button
            with dpg.collapsing_header(label="Train", default_open=True):
                with dpg.group(horizontal=True):
                    dpg.add_text("Train: ")

                    def callback_train(sender, app_data):
                        if self.require_training:
                            self.require_training = False
                            dpg.configure_item("_button_train", label="start")
                        else:
                            self.require_training = True
                            dpg.configure_item("_button_train", label="stop")

                    dpg.add_button(
                        label="start", tag="_button_train", callback=callback_train
                    )
                    dpg.bind_item_theme("_button_train", theme_button)

                    def callback_reset(sender, app_data):
                        self.trainer.reset()
                        self.need_update = True
                        self.require_training = False
                        self.step = 0
                        dpg.configure_item("_button_train", label="start")

                    dpg.add_button(
                        label="reset", tag="_button_reset", callback=callback_reset
                    )
                    dpg.bind_item_theme("_button_reset", theme_button)

                    # save ckpt
                    with dpg.group(horizontal=True):
                        dpg.add_text("Checkpoint: ")

                        def callback_save_ckpt(sender, app_data):
                            self.trainer.save_model()
                            dpg.set_value("_log_ckpt", "saved")
                            time.sleep(1)
                            dpg.set_value("_log_ckpt", "")

                        dpg.add_button(
                            label="save",
                            tag="_button_save_ckpt",
                            callback=callback_save_ckpt,
                        )
                        dpg.bind_item_theme("_button_save_ckpt", theme_button)

                        dpg.add_text("", tag="_log_ckpt")

                with dpg.group(horizontal=True):
                    dpg.add_text("", tag="_log_train_log")
                with dpg.group(horizontal=True):
                    dpg.add_text("", tag="_log_train_loss")
                with dpg.group(horizontal=True):
                    dpg.add_text("", tag="_log_train_performance")

            # rendering options
            with dpg.collapsing_header(label="Options", default_open=True):
                # dynamic rendering resolution
                with dpg.group(horizontal=True):

                    def callback_set_dynamic_resolution(sender, app_data):
                        if self.dynamic_resolution:
                            self.dynamic_resolution = False
                            self.downscale = 1
                        else:
                            self.dynamic_resolution = True
                        self.need_update = True

                    dpg.add_checkbox(
                        label="dynamic resolution",
                        default_value=self.dynamic_resolution,
                        callback=callback_set_dynamic_resolution,
                    )
                    dpg.add_text(f"{self.H}x{self.W}", tag="_log_resolution")

                # reset camera view
                def callback_init_camera_pose(sender, app_data):
                    self.cam.to_initial_pose()
                    dpg.set_value("_log_camera_pose", str(self.cam.pose[:3, 3]))
                    self.need_update = True

                dpg.add_button(
                    label="init camera pose",
                    tag="_button_init_camera_pose",
                    callback=callback_init_camera_pose,
                )

                dpg.bind_item_theme("_button_init_camera_pose", theme_button)

                # save screen shot
                with dpg.group(horizontal=True):

                    def transform_2_pil(image_date):
                        return Image.fromarray((image_date * 255).astype(np.uint8))

                    self.screenshot_id = 0

                    def callback_save_screenshot(sender, app_data):
                        outputs = self.trainer.render(
                            self.cam.pose,
                            self.cam.intrinsics,
                            self.H,
                            self.W,
                            self.downscale,
                            -1,
                            self.rendering_target_id,
                        )
                        save_path = f"{self.trainer.record_path}/screenshot/{str(self.screenshot_id)}"
                        os.makedirs(save_path)

                        rgb = transform_2_pil(outputs["rgb"])
                        rgb.save(f"{save_path}/rgb.png")

                        depth = transform_2_pil(outputs["depth"])
                        depth.save(f"{save_path}/depth.png")

                        acc = transform_2_pil(outputs["acc"])
                        acc.save(f"{save_path}/acc.png")

                        depth_unc = transform_2_pil(outputs["depth_uncertainty"])
                        depth_unc.save(f"{save_path}/depth_unc.png")

                        semantic = transform_2_pil(outputs["semantic"])
                        semantic.save(f"{save_path}/semantic.png")

                        semantic_unc = transform_2_pil(outputs["semantic_uncertainty"])
                        semantic_unc.save(f"{save_path}/semantic_unc.png")

                        trajectory = transform_2_pil(outputs["trajectory"])
                        trajectory.save(f"{save_path}/trajectory.png")
                        self.screenshot_id += 1

                        dpg.set_value("_log_screenshot", "saved")
                        time.sleep(0.5)
                        dpg.set_value("_log_screenshot", "")

                    dpg.add_button(
                        label="save screenshot",
                        tag="_button_save_screenshot",
                        callback=callback_save_screenshot,
                    )
                    dpg.add_text("", tag="_log_screenshot")

                    dpg.bind_item_theme("_button_save_screenshot", theme_button)

                # render only target class
                def callback_render_target(sender, app_data):
                    self.rendering_target = app_data

                    if self.rendering_target != "all":
                        self.rendering_target_id = [
                            LABELS["shapenet"][self.rendering_target]["id"]
                        ]
                    else:
                        self.rendering_target_id = []

                    self.need_update = True

                dpg.add_combo(
                    ("all", "car", "table", "chair", "camera", "airplane", "sofa"),
                    label="target",
                    default_value=self.rendering_target,
                    callback=callback_render_target,
                )

                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(
                    ("rgb", "depth", "acc"),
                    label="mode",
                    default_value=self.mode,
                    callback=callback_change_mode,
                )

                with dpg.group(horizontal=True):
                    dpg.add_text("Camera Pose:")
                    dpg.add_text(str(self.cam.pose[:3, 3]), tag="_log_camera_pose")

        ### register camera handler

        def callback_camera_drag_rotate(sender, app_data):
            if not dpg.is_item_focused("_main_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            dpg.set_value("_log_camera_pose", str(self.cam.pose[:3, 3]))
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_main_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            dpg.set_value("_log_camera_pose", str(self.cam.pose[:3, 3]))
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_main_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            dpg.set_value("_log_camera_pose", str(self.cam.pose[:3, 3]))
            self.need_update = True

        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

        dpg.create_viewport(
            title="Semantic-Targeted Active Implicit Reconstruction",
            width=3 * self.W,
            height=2 * self.H,
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_main_window", theme_no_padding)
        dpg.bind_item_theme("_depth_unc_window", theme_no_padding)
        dpg.bind_item_theme("_semantic_window", theme_no_padding)
        dpg.bind_item_theme("_trajectory_window", theme_no_padding)

        dpg.setup_dearpygui()
        dpg.show_viewport()

    @property
    def planning_permission(self):
        return (
            (self.require_planning or self.start_mission)
            and self.render_done
            and self.training_done
        )

    def plan_step(self):
        if self.planning_step < self.planning_budget:
            steps = self.trainer.planning()
            self.need_update = True
            self.planning_step += steps

            self.require_planning = False
            dpg.set_value(
                "_log_plan_step",
                f"planning step: {self.planning_step}",
            )
            if (
                self.trainer.map_update_for_planning
                and self.planning_step <= self.planning_budget
            ):
                self.require_training = True
        else:
            self.require_planning = False
            self.require_training = False
            self.start_mission = False
            self.trainer.save_experiment()
            dpg.set_value(
                "_log_plan_step",
                f"planning step: budget reached!!!",
            )

    def update_step(self):
        self.training_done = False
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        starter.record()

        outputs, iteration = self.trainer.update(self.train_steps)

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender) / self.train_steps

        self.step = iteration
        self.need_update = True

        if outputs is not None:
            dpg.set_value("_log_train_time", f"{t:.4f}ms ({int(1000/t)} FPS)")
            dpg.set_value(
                "_log_train_log", f"step = {self.step: 4d} (+{self.train_steps: 2d})"
            )
            dpg.set_value(
                "_log_train_loss",
                f'rgb_loss = {outputs["rgb_loss"]:.4f}, semantic_loss = {outputs["semantic_loss"]:.4f}',
            )
            dpg.set_value("_log_train_performance", f'psnr = {outputs["psnr"]:.4f}')

        elif outputs is None:
            self.training_done = True
            self.require_training = False
            self.trainer.update_end()

    def render_step(self):
        if self.need_update:
            self.render_done = False
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            starter.record()

            outputs = self.trainer.render(
                self.cam.pose,
                self.cam.intrinsics,
                self.H,
                self.W,
                self.downscale,
                -1,
                self.rendering_target_id,
            )

            ender.record()
            torch.cuda.synchronize()
            t = starter.elapsed_time(ender)

            # update dynamic resolution
            if self.dynamic_resolution:
                # max allowed infer time per-frame is 200 ms
                full_t = t / (self.downscale**2)
                downscale = min(1, max(1 / 4, math.sqrt(200 / full_t)))
                if downscale > self.downscale * 1.2 or downscale < self.downscale * 0.8:
                    self.downscale = downscale

            if self.mode == "rgb":
                self.render_buffer_main = outputs["rgb"]
            elif self.mode == "depth":
                self.render_buffer_main = outputs["depth"]
            elif self.mode == "acc":
                self.render_buffer_main = outputs["acc"]
            self.render_buffer_depth_unc = outputs["depth_uncertainty"]
            self.render_buffer_semantic = outputs["semantic"]
            self.render_buffer_semantic_unc = outputs["semantic_uncertainty"]
            self.render_buffer_trajectory = outputs["trajectory"]

            dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS)")
            dpg.set_value(
                "_log_resolution",
                f"{int(self.downscale * self.H)}x{int(self.downscale * self.W)}",
            )
            dpg.set_value("_main_texture", self.render_buffer_main)
            dpg.set_value("_depth_unc_texture", self.render_buffer_depth_unc)
            dpg.set_value("_semantic_texture", self.render_buffer_semantic)
            dpg.set_value("_trajectory_texture", self.render_buffer_trajectory)

            dpg.render_dearpygui_frame()

            self.render_done = True
            self.need_update = False

    def _planning(self):
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
            if self.planning_permission:
                self.plan_step()
            if self.require_training:
                self.update_step()
            self.render_step()

    def _test(self):
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
            if self.require_training:
                self.update_step()
            self.render_step()

    def start(self):
        if self.planning_mode:
            self._planning()
        else:
            self._test()
