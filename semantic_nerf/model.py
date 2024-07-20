import torch
import torch.nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical
import time
from tools.utils import *
from tools.mesh import MeshExtractor
from .base import *
import pdb


class SemanticNeRF(torch.nn.Module):
    def __init__(self, args, device):
        super(SemanticNeRF, self).__init__()
        self.device = device

        # volume grid setup
        self.rgb_dim = args.rgb_dim
        self.density_dim = args.density_dim
        self.semantic_dim = args.semantic_dim
        self.class_num = args.class_num
        self.near_far = args.near_far
        self.n_samples_max = args.n_samples_max
        self.step_ratio = args.step_ratio
        self.aabb = args.aabb
        self.n_voxel = args.n_voxel
        self.occupancy_thres = args.occupancy_thres
        self.mesh_resolution = args.mesh_resolution
        self.level_set = args.level_set
        self.radius = args.radius  # camera radius
        self.max_env = np.max(self.aabb[1])

        # shading setup
        self.shadingmode_rgb = args.shadingmode_rgb
        self.shadingmode_density = args.shadingmode_density
        self.shadingmode_semantic = args.shadingmode_semantic
        self.pos_pe = args.pos_pe
        self.view_pe = args.view_pe
        self.fea_pe = args.fea_pe
        self.feature_rgb = args.feature_rgb
        self.feature_semantic = args.feature_semantic
        self.use_xyz = args.use_xyz
        self.use_view = args.use_view
        self.density_shift = args.density_shift
        self.raymarch_weight_thres = args.raymarch_weight_thres
        self.fea2dense_act = args.fea2dense_act
        self.use_gradientscaler = args.use_gradientscaler
        self.sampling_random_level = args.sampling_random_level
        self.softmax = torch.nn.Softmax(dim=-1)

        # initialize model
        self.rendermodule_rgb = self.init_rgb_render_func()
        self.rendermodule_density = self.init_density_render_func()
        self.rendermodule_semantic = self.init_semantic_render_func()

        self.update_grid_config(self.aabb, self.n_voxel)
        self.init_volume()

    def init_rgb_render_func(self):
        if self.shadingmode_rgb == "MLP":
            rendermodule = RGBRenderMLP(
                self.rgb_dim,
                self.pos_pe,
                self.view_pe,
                self.fea_pe,
                self.feature_rgb,
                self.use_xyz,
                self.use_view,
            ).to(self.device)
        elif self.shadingmode_rgb == "SH":
            rendermodule = SHRender
        elif self.shadingmode_rgb == "IF":
            self.rgb_dim = 3
            rendermodule = RGBRender
        else:
            print("Unrecognized shading module")
            exit()
        print(f"use {self.shadingmode_rgb} for rgb")
        return rendermodule

    def init_density_render_func(self):
        if self.shadingmode_density == "MLP":
            rendermodule = DensityRenderMLP(
                self.density_dim,
                self.pos_pe,
            ).to(self.device)
        elif self.shadingmode_density == "IF":
            self.density_dim = 1
            rendermodule = DensityRender
        else:
            print("Unrecognized shading module")
            exit()
        print(f"use {self.shadingmode_density} for density")
        return rendermodule

    def init_semantic_render_func(self):
        if self.shadingmode_semantic == "MLP":
            rendermodule = SemanticRenderMLP(
                self.semantic_dim, self.class_num, self.feature_semantic
            ).to(self.device)
        elif self.shadingmode_semantic == "IF":
            self.semantic_dim = self.class_num
            rendermodule = SemanticRender
        else:
            print("Unrecognized shading module")
            exit()
        print(f"use {self.shadingmode_semantic} for semantic")
        return rendermodule

    @torch.no_grad()
    def update_grid_config(self, aabb, n_voxel, grid_size=None):
        self.aabb = torch.tensor(aabb).to(self.device)
        grid_size = n_to_reso(n_voxel, self.aabb) if grid_size is None else grid_size
        self.grid_size = torch.LongTensor(grid_size).to(self.device)

        print("aabb", self.aabb.view(-1))
        print("grid size", grid_size)

        self.aabb_size = self.aabb[1] - self.aabb[0]
        self.invaabb_size = 2.0 / self.aabb_size
        self.units = self.aabb_size / (self.grid_size - 1)
        self.step_size = torch.mean(self.units) * self.step_ratio
        self.aabb_diag = torch.sqrt(torch.sum(torch.square(self.aabb_size)))
        n_samples = int((self.aabb_diag / self.step_size).item()) + 1
        self.n_samples = min(self.n_samples_max, n_samples)

        print("sampling step size: ", self.step_size)
        print("sampling number: ", self.n_samples)

    @torch.no_grad()
    def extract_bbox(self, target_class_id=None):
        occupancy_field, dense_xyz = self.get_dense_occupancy(
            target_class_id=target_class_id
        )
        occ_mask = occupancy_field > self.occupancy_thres
        valid_dense_xyz = dense_xyz[occ_mask].permute(1, 0)  # (3, N)

        min_x = torch.min(valid_dense_xyz[0]).item()
        max_x = torch.max(valid_dense_xyz[0]).item()
        min_y = torch.min(valid_dense_xyz[1]).item()
        max_y = torch.max(valid_dense_xyz[1]).item()
        min_z = torch.min(valid_dense_xyz[2]).item()
        max_z = torch.max(valid_dense_xyz[2]).item()

        new_aabb = (
            [min_x, min_y, min_z],
            [max_x, max_y, max_z],
        )

        return new_aabb

    def query_occupancy(self, xyz_query, target_class_id):
        xyz_query = self.normalize_coord(xyz_query)  # (1, N, 3)
        density_feature = self.compute_density_feature(xyz_query)
        occupancy = self.feature2occ(xyz_query, density_feature)  # (1 ,N)
        occupancy = occupancy.view(-1)  # (N)

        # filter occupancy field by semantic information
        if len(target_class_id) > 0:
            semantic_features = self.compute_semantic_feature(xyz_query)
            semantics = self.rendermodule_semantic(xyz_query, semantic_features)
            semantic_prob = self.softmax(semantics)
            semantic_label = torch.argmax(semantic_prob, dim=-1)
            semantic_label = semantic_label.view(-1)  # (N)

            semantic_mask = sum(semantic_label == i for i in target_class_id).bool()
            semantic_mask = semantic_mask.long()
            occupancy *= semantic_mask

        return occupancy  # (N)

    @torch.no_grad()
    def get_dense_occupancy(
        self, grid_size=None, target_class_id=None, extract_aabb=None
    ):
        grid_size = self.grid_size if grid_size is None else grid_size
        aabb = self.aabb if extract_aabb is None else extract_aabb
        samples = torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, grid_size[0]),
                torch.linspace(0, 1, grid_size[1]),
                torch.linspace(0, 1, grid_size[2]),
            ),
            -1,
        ).to(
            self.device
        )  # (*grid_size, 3)

        dense_xyz = aabb[0] * (1 - samples) + aabb[1] * samples
        dense_xyz = dense_xyz.view(1, -1, 3)  # (1, N, 3)
        occupancy = self.query_occupancy(dense_xyz, target_class_id)
        dense_xyz = dense_xyz.squeeze(0)

        return (
            occupancy,  # (N)
            dense_xyz,  # (N, 3)
        )

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled - self.aabb[0]) * self.invaabb_size - 1

    def feature2density(self, density_features):
        if self.fea2dense_act == "softplus":
            return F.softplus(density_features + self.density_shift)
        elif self.fea2dense_act == "relu":
            return F.relu(density_features)
        elif self.fea2dense_act == "sigmoid":
            return F.sigmoid(density_features)

    def feature2occ(self, pts, density_feature):
        feature = self.rendermodule_density(pts, density_feature)
        sigma = self.feature2density(feature)
        occ = 1 - torch.exp(-sigma)
        return occ.squeeze(-1)

    def save_mesh(self, path, target_class_id=[]):
        extractor = MeshExtractor(
            self,
            device=self.device,
            resolution0=self.mesh_resolution,
            target_class_id=target_class_id,
        )

        mesh = extractor.generate_mesh(aabb=self.aabb)
        mesh.export(path)

    def save_model(self, path):
        kwargs = self.get_kwargs()
        ckpt = {"kwargs": kwargs, "state_dict": self.state_dict()}
        torch.save(ckpt, path)

    def load_model(self, ckpt):
        print(ckpt["kwargs"])
        self.load_state_dict(ckpt["state_dict"])

    def sample_ray(self, rays_o, rays_d, is_train=True, n_samples=-1, depth=None):
        if n_samples > 0:
            n_samples = min(self.n_samples_max, n_samples)
            stepsize = self.step_size * self.n_samples / n_samples
        else:
            n_samples = self.n_samples
            stepsize = self.step_size

        near, far = self.near_far
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec

        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
        rng = torch.arange(n_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2], 1)
            rng += torch.rand_like(rng[:, [0]])
        step = stepsize * rng.to(rays_o.device)
        interpx = t_min[..., None] + step

        rays_pts = (
            rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        )  # (B, 1, 3)

        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(
            dim=-1
        )
        mask_outradius = torch.linalg.norm(rays_pts, dim=-1) > self.max_env  # radius
        mask_outbbox += mask_outradius
        return rays_pts, interpx, ~mask_outbbox

    @torch.no_grad()
    def filtering_rays(
        self,
        all_rays,
        chunk=10240 * 5,
        bbox_only=True,
    ):
        print("========> filtering rays ...")
        tt = time.time()

        N = torch.tensor(all_rays.shape[:-1]).prod()

        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk].to(self.device)

            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            if bbox_only:
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.aabb[1] - rays_o) / vec
                rate_b = (self.aabb[0] - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(
                    -1
                )  # .clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(
                    -1
                )  # .clamp(min=near, max=far)
                mask_inbbox = t_max > t_min

            mask_filtered.append(mask_inbbox.cpu())

        mask_filtered = torch.cat(mask_filtered).view(all_rays.shape[:-1])

        print(
            f"Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {torch.sum(mask_filtered) / N}"
        )
        return mask_filtered

    def forward(
        self,
        rays_chunk,
        require_class=[],
        white_bg=True,
        is_train=False,
        n_samples=-1,
        use_depth=False,
    ):
        # sample points
        viewdirs = rays_chunk[:, 3:6]
        depths = rays_chunk[:, -1]

        xyz_sampled, z_vals, ray_valid = self.sample_ray(
            rays_chunk[:, :3],
            viewdirs,
            is_train=is_train,
            n_samples=n_samples,
            # depth=depths,
        )

        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)
        xyz_sampled = self.normalize_coord(xyz_sampled)  # (B, N, 3)
        occ = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)  # (B, N)
        rgb = torch.zeros(
            (*xyz_sampled.shape[:2], 3), device=xyz_sampled.device
        )  # (B, N, 3)
        semantic = torch.zeros(
            (*xyz_sampled.shape[:2], self.class_num), device=xyz_sampled.device
        )  # (B, N, C)

        if ray_valid.any():
            density_feature = self.compute_density_feature(xyz_sampled[ray_valid])
            occ_valid = self.feature2occ(xyz_sampled[ray_valid], density_feature)
            occ[ray_valid] = occ_valid

            if self.use_gradientscaler:
                norm_ray_dist = z_vals / self.radius
                occ, _ = GradientScalerOcc.apply(occ, norm_ray_dist)

            ray_valid = occ > self.occupancy_thres  # ignore empty space
            rgb_features = self.compute_rgb_feature(xyz_sampled[ray_valid])
            semantic_features = self.compute_semantic_feature(xyz_sampled[ray_valid])

            valid_rgbs = self.rendermodule_rgb(
                xyz_sampled[ray_valid], viewdirs[ray_valid], rgb_features
            )

            valid_semantics = self.rendermodule_semantic(
                xyz_sampled[ray_valid], semantic_features
            )
            rgb[ray_valid] = valid_rgbs
            semantic[ray_valid] = valid_semantics

            if not is_train:
                _semantic_label = torch.argmax(semantic, dim=-1)
                delete_mask = _semantic_label == 0
                if len(require_class) > 0:
                    label_mask = sum(_semantic_label == i for i in require_class).bool()
                    delete_mask += ~label_mask
                    delete_mask = delete_mask.bool()
                occ[delete_mask] = 0

        valid_pixel = torch.sum(ray_valid, dim=-1).bool()
        weight, trans = occ2weight(occ)

        max_unknown_map = torch.zeros(
            xyz_sampled.shape[0], device=xyz_sampled.device
        )  # (B)
        behind_surface_mask = (trans < 0.001) * ray_valid
        if behind_surface_mask.any():
            valid_unknow_mask = torch.sum(behind_surface_mask, dim=-1).bool()
            max_unknown_map[valid_unknow_mask] = (
                torch.sum(torch.abs(occ - 0.5) * behind_surface_mask, dim=-1)[
                    valid_unknow_mask
                ]
                / torch.sum(behind_surface_mask, dim=-1)[valid_unknow_mask]
            )

        acc_map = torch.sum(weight, -1).clamp(0, 1)

        # rgb map
        rgb_map = torch.sum(weight[..., None] * rgb, -2)
        if white_bg and not is_train:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])
        rgb_map = rgb_map.clamp(0, 1)

        weight_no_grad = weight.detach()
        trans_no_grad = trans.detach().clone()

        # semantic map
        semantic_map = torch.sum((weight_no_grad[..., None] * semantic), -2)
        semantic_map = self.softmax(semantic_map)

        # detph map
        depth_map = torch.sum((weight * z_vals), -1)
        depth_map = depth_map.clamp(0, self.near_far[1])

        # uncertainty rendering
        depth_unc_map = torch.zeros(
            xyz_sampled.shape[0], device=xyz_sampled.device
        )  # B
        semantic_unc_map = torch.zeros(
            xyz_sampled.shape[0], device=xyz_sampled.device
        )  # B

        object_mask = acc_map.detach() > 0.01
        valid_pixel = valid_pixel * object_mask

        per_point_geometric_entropy = Bernoulli(probs=occ).entropy() / torch.log(
            torch.tensor(2)
        )
        depth_unc_map[valid_pixel] = (
            (
                1
                - torch.exp(
                    -torch.sum(
                        per_point_geometric_entropy * trans_no_grad,
                        dim=-1,
                    )
                )
            )
        )[valid_pixel]

        # semantic_unc_map[valid_pixel] = (
        #     Categorical(probs=semantic_map + 1e-8).entropy()
        #     / torch.log(torch.tensor(self.class_num))
        # )[valid_pixel]

        return (
            rgb_map,
            depth_map,
            semantic_map,
            depth_unc_map,
            semantic_unc_map,
            acc_map,
            max_unknown_map,
        )

    def init_volume(self):
        raise NotImplementedError("init_volume is not implemented")

    def compute_density_feature(self, xyz_sampled):
        raise NotImplementedError("compute_density_feature is not implemented")

    def compute_rgb_feature(self, xyz_sampled):
        raise NotImplementedError("compute_rgb_feature is not implemented")

    def compute_semantic_feature(self, xyz_sampled):
        raise NotImplementedError("compute_semantic_feature is not implemented")

    def get_kwargs(self):
        raise NotImplementedError("get_kwargs is not implemented")

    @torch.no_grad()
    def up_sampling(self):
        raise NotImplementedError("up_sampling is not implemented")

    @torch.no_grad()
    def upsample_volume(self):
        raise NotImplementedError("upsample_volume is not implemented")

    @torch.no_grad()
    def shrink_volume(self):
        raise NotImplementedError("shrink_volume is not implemented")

    def get_optparam_groups(self):
        raise NotImplementedError("get_optparam_groups is not implemented")

    def density_l1(self):
        raise NotImplementedError("density_l1 is not implemented")

    def tv_loss_density(self, reg):
        raise NotImplementedError("tv_loss_density is not implemented")

    def tv_loss_rgb(self, reg):
        raise NotImplementedError("tv_loss_rgb is not implemented")

    def tv_loss_semantic(self, reg):
        raise NotImplementedError("tv_loss_semantic is not implemented")

    def me_loss_semantic(self):
        raise NotImplementedError("me_loss_semantic is not implemented")

    def me_loss_density(self):
        raise NotImplementedError("me_loss_density is not implemented")
