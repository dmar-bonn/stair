from tools.utils import *
from tools.mesh import MeshExtractor
from semantic_nerf import get_model
from ..mapper_base import Mapper


class ImplicitMapper(Mapper):
    def __init__(self, args, device):
        super(ImplicitMapper, self).__init__(args)
        self.args = args
        self.device = device
        self.renderer = BatchRenderer

        # training setup
        self.batch_size = args.batch_size
        self.white_bg = args.white_bg
        self.require_depth = args.require_depth
        self.num_history = 0
        self.sampling_count = np.array([])
        self.active_buffer_len = args.active_buffer_len
        self.buffer_sample_ratio = args.buffer_sample_ratio

        # main loss function
        self.rgb_loss_fnc = nn.L1Loss()
        self.semantic_loss_fnc = nn.NLLLoss(reduction="none")

        self.init_traning_setup()

    def load_training_args(self, is_fine_stage):
        if is_fine_stage:
            training_args = self.args.fine_stage
        else:
            training_args = self.args.coarse_stage

        self.args.update(training_args)

    def init_traning_setup(self, fine_stage=False):
        print("---------- initialize mapper ----------")
        self.fine_stage = fine_stage
        self.load_training_args(fine_stage)

        if fine_stage:
            print("use simple sampler")
            self.training_sampler = SimpleSampler(
                self.all_rays.shape[0], self.batch_size
            )

        # nerf model
        self.model = get_model(self.args, self.device)

        # optimizer
        self.lr_volume = self.args.lr_volume
        self.lr_mlp = self.args.lr_mlp

        self.rgb_reg_weight = self.args.rgb_reg_weight
        self.semantic_reg_weight = self.args.semantic_reg_weight
        self.max_unknown_weight = self.args.max_unknown_weight
        self.density_unc_weight = self.args.density_unc_weight
        self.depth_weight = self.args.depth_weight
        self.tv_weight_semantic = self.args.tv_weight_semantic

        grad_vars, param_types = self.model.get_optparam_groups(
            self.lr_volume, self.lr_mlp
        )

        self.lr_factor = []
        for param_type in param_types:
            if param_type == "spatial":
                self.lr_factor.append(self.args.lr_factor_spatial)
            elif param_type == "mlp":
                self.lr_factor.append(self.args.lr_factor_mlp)
            else:
                RuntimeError
        print(self.lr_factor)

        self.optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
        self.iteration = 0
        torch.cuda.empty_cache()

    def update_training_setup(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = param_group["lr"] * self.lr_factor[i]

    def update_map(self, train_step):
        rgb_loss_total = 0
        semantic_loss_total = 0
        psnr_total = 0

        for _ in range(train_step):
            self.iteration += 1
            ray_idx = self.training_sampler.nextids()
            self.sampling_count[ray_idx.numpy()] += 1
            rgb_loss, semantic_loss, psnr = self.train_step(ray_idx)
            self.update_training_setup()
            rgb_loss_total += rgb_loss
            semantic_loss_total += semantic_loss
            psnr_total += psnr

        outputs = {
            "rgb_loss": rgb_loss_total / train_step,
            "semantic_loss": semantic_loss_total / train_step,
            "psnr": psnr_total / train_step,
        }
        return outputs, self.iteration

    @torch.no_grad()
    def render_view(
        self,
        pose,
        intrinsics,
        H,
        W,
        downscale,
        n_samples,
        rendering_target_id=[],
        planning=False,
    ):
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).to(self.device)
        if len(pose.shape) == 2:
            pose = pose.unsqueeze(0)
            B = 1
        else:
            B = pose.shape[0]
        rays = get_rays_gui(pose, intrinsics, rH, rW)

        (
            rgb_map,
            semantic_map,
            depth_map,
            depth_unc_map,
            semantic_unc_map,
            acc_map,
            _,
        ) = self.renderer(
            rays,
            self.model,
            require_class=rendering_target_id,
            chunk=self.batch_size,
            n_samples=n_samples,
            white_bg=self.white_bg,
            device=self.device,
        )  # (B*rH*rW, ...)

        rgb_map, semantic_map, depth_map, depth_unc_map, semantic_unc_map, acc_map = (
            rgb_map.reshape(B, rH, rW, 3),
            semantic_map.reshape(B, rH, rW, self.class_num),
            depth_map.reshape(B, rH, rW),
            depth_unc_map.reshape(B, rH, rW),
            semantic_unc_map.reshape(B, rH, rW),
            acc_map.reshape(B, rH, rW),
        )

        if downscale != 1:
            depth_unc_map = F.interpolate(
                depth_unc_map.unsqueeze(0), size=(H, W), mode="bilinear"
            )[0]
            semantic_unc_map = F.interpolate(
                semantic_unc_map.unsqueeze(0), size=(H, W), mode="bilinear"
            )[0]
            depth_map = F.interpolate(
                depth_map.unsqueeze(0), size=(H, W), mode="bilinear"
            )[0]
            acc_map = F.interpolate(acc_map.unsqueeze(0), size=(H, W), mode="bilinear")[
                0
            ]
            object_ratio = torch.sum(acc_map.view(B, -1) > 0.1, dim=-1) / (rH * rW)

            rgb_map = (
                F.interpolate(
                    rgb_map.permute(0, 3, 1, 2),
                    size=(H, W),
                    mode="bilinear",
                )
                .permute(0, 2, 3, 1)
                .contiguous()
            )
            semantic_map = (
                F.interpolate(
                    semantic_map.permute(0, 3, 1, 2),
                    size=(H, W),
                    mode="nearest",
                )
                .permute(0, 2, 3, 1)
                .contiguous()
            )

        depth_unc_map = F.avg_pool2d(
            depth_unc_map.unsqueeze(1), kernel_size=5, padding=5 // 2, stride=1
        ).squeeze(1)

        semantic_map = torch.argmax(semantic_map, dim=-1)

        if planning:  # for planning
            outputs = {
                "depth_uncertainty": depth_unc_map.cpu().numpy(),
                "semantic_uncertainty": semantic_unc_map.cpu().numpy(),
                "semantic": semantic_map.cpu().numpy(),
                "object_ratio": object_ratio.cpu().numpy(),
            }
        else:  # for visualization
            outputs = {
                "rgb": rgb_map[0].cpu().numpy(),
                "semantic": label2color(semantic_map[0].cpu().numpy(), "shapenet"),
                "depth": visualize_depth_numpy(
                    depth_map[0].cpu().numpy(), self.near_far
                ),
                "depth_uncertainty": visualize_depth_unc_numpy(
                    depth_unc_map[0].cpu().numpy()
                ),
                "semantic_uncertainty": visualize_semantic_unc_numpy(
                    semantic_unc_map[0].cpu().numpy()
                ),
                "acc": visualize_acc_numpy(
                    acc_map[0].cpu().numpy(),
                ),
            }
        return outputs

    def update_measurement_buffer(self, buffer, use_replay_buffer=True):
        (
            all_rays,
            all_rgbs,
            all_semantics,
            all_semantics_unc,
            all_depths,
            all_semantics_mask,
            all_semantics_gt,
            _,
            _,
        ) = buffer

        self.all_rays = all_rays
        self.all_rgbs = all_rgbs
        self.all_semantics = all_semantics
        self.all_semantics_unc = all_semantics_unc
        self.all_depths = all_depths
        self.all_semantics_mask = all_semantics_mask
        self.all_semantics_gt = all_semantics_gt

        self.background_mask = self.all_semantics_gt == 0
        self.object_mask = self.all_semantics_gt != 0

        self.all_rgbs[self.background_mask] = (
            0 * self.all_rgbs[self.background_mask] + 1.0
        )
        self.all_depths[self.background_mask] = 0

        # setup sampler for training
        self.num_rays = self.all_rays.shape[0]
        num_new = self.num_rays - self.num_history
        self.sampling_count = np.append(self.sampling_count, np.ones(num_new))

        if self.num_history > 0:
            sample_weight = 1 / self.sampling_count[: self.num_history]
            sample_weight = sample_weight / np.sum(sample_weight)
            active_buffer = torch.tensor(
                np.random.choice(
                    self.num_history,
                    np.min((self.num_history, self.active_buffer_len)),
                    replace=False,
                    p=sample_weight,
                )
            )  # active ray index in buffer for this training session
        else:
            active_buffer = torch.tensor([])
        if use_replay_buffer:
            self.training_sampler = ReplaySampler(
                active_buffer,
                num_new,
                self.batch_size,
                self.buffer_sample_ratio,
                offset=self.num_history,
            )
        else:
            self.training_sampler = SimpleSampler(
                self.all_rays.shape[0], self.batch_size
            )

        self.num_history = self.num_rays

    def train_step(self, ray_idx):
        (
            rays_train,
            semantic_train,
            rgb_train,
            valid_semantic_mask_train,
            depth_train,
            bg_mask_train,
            obj_mask_train,
        ) = (
            self.all_rays[ray_idx],
            self.all_semantics[ray_idx],
            self.all_rgbs[ray_idx],
            self.all_semantics_mask[ray_idx],
            self.all_depths[ray_idx],
            self.background_mask[ray_idx],
            self.object_mask[ray_idx],
        )

        if self.require_depth:
            rays_train = torch.cat((rays_train, depth_train.unsqueeze(-1)), dim=-1)

        (
            rgb_map,
            semantic_map,
            depth_map,
            depth_unc_map,
            semantic_unc_map,
            acc_map,
            unknown_map,
        ) = self.renderer(
            rays_train,
            self.model,
            chunk=self.batch_size,
            n_samples=-1,
            white_bg=self.white_bg,
            device=self.device,
            is_train=True,
            use_depth=self.require_depth,
        )

        rgb_train = rgb_train.to(self.device)
        depth_train = depth_train.to(self.device)
        semantic_train = semantic_train.to(self.device)

        bg_mask_train = bg_mask_train.to(self.device).long()
        obj_mask_train = obj_mask_train.to(self.device).long()
        valid_semantic_mask_train = valid_semantic_mask_train.to(self.device).long()

        rgb_loss = 0
        if obj_mask_train.any():
            rgb_loss_raw = torch.sum(
                torch.mean(torch.square(rgb_map - rgb_train), dim=-1) * obj_mask_train
            ) / torch.sum(obj_mask_train)
            rgb_loss += rgb_loss_raw

        # rgb regularization loss
        if self.max_unknown_weight > 0 and obj_mask_train.any():
            max_unknown_loss = torch.sum(unknown_map * obj_mask_train) / torch.sum(
                obj_mask_train
            )
            max_unknown_loss *= self.max_unknown_weight
            rgb_loss += max_unknown_loss

        if self.require_depth and obj_mask_train.any():
            depth_loss = torch.sum(
                torch.abs(depth_map - depth_train) * obj_mask_train
            ) / torch.sum(obj_mask_train)
            depth_loss *= self.depth_weight
            rgb_loss += depth_loss

        if self.density_unc_weight > 0:
            density_unc_loss = torch.mean(depth_unc_map)
            density_unc_loss *= self.density_unc_weight
            rgb_loss += density_unc_loss

        if self.rgb_reg_weight > 0:
            if bg_mask_train.any():
                free_space_loss = torch.sum(
                    torch.square(acc_map - 0) * bg_mask_train
                ) / torch.sum(bg_mask_train)
                free_space_loss *= self.rgb_reg_weight
                rgb_loss += free_space_loss

        # semantic regularization loss
        semantic_loss = 0
        if self.tv_weight_semantic > 0:
            tv_semantic_loss = self.model.tv_loss_semantic()
            semantic_loss += self.tv_weight_semantic * tv_semantic_loss

        pred_bg_mask = acc_map.detach() < 0.2
        pred_bg_mask = pred_bg_mask.long()
        pred_obj_mask = acc_map.detach() > 0.8
        pred_obj_mask = pred_obj_mask.long()
        pred_bg_prob = semantic_map[:, 0]  # B

        if self.semantic_reg_weight > 0:
            if pred_bg_mask.any():
                pred_bg_loss = torch.sum(
                    torch.square(pred_bg_prob - 1) * pred_bg_mask
                ) / torch.sum(pred_bg_mask)
                semantic_loss += self.semantic_reg_weight * pred_bg_loss

            if pred_obj_mask.any():
                pred_obj_loss = torch.sum(
                    torch.square(pred_bg_prob - 0) * pred_obj_mask
                ) / torch.sum(pred_obj_mask)
                semantic_loss += self.semantic_reg_weight * pred_obj_loss

        semantic_train_label = torch.argmax(semantic_train, dim=-1)
        if valid_semantic_mask_train.any():
            semantic_loss_raw = torch.sum(
                self.semantic_loss_fnc(
                    torch.log(semantic_map + 1e-8), semantic_train_label
                )
                * valid_semantic_mask_train
            ) / torch.sum(valid_semantic_mask_train)
            semantic_loss += semantic_loss_raw

        self.optimizer.zero_grad()
        rgb_loss.backward()

        if semantic_loss.requires_grad:
            semantic_loss.backward()

        self.optimizer.step()
        semantic_loss_item = semantic_loss.detach().item()
        rgb_loss_item = rgb_loss.detach().item()

        rgb_mse = torch.mean((rgb_map.detach() - rgb_train) ** 2).cpu()
        psnr = -10 * torch.log10(rgb_mse).item()

        return (rgb_loss_item, semantic_loss_item, psnr)

    def save_model(self, model_file):
        self.model.save_model(model_file)

    def save_mesh(self, mesh_file, target_class_id):
        self.model.save_mesh(mesh_file, target_class_id)
