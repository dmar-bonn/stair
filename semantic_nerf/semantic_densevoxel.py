from .model import *


class DenseGrid(torch.nn.Module):
    def __init__(self, channels, grid_size, init_val):
        super(DenseGrid, self).__init__()
        self.channels = channels
        self.grid = torch.nn.Parameter(init_val * torch.ones([1, channels, *grid_size]))

    def forward(self, xyz):
        """
        xyz: global coordinates to query
        """
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)
        out = F.grid_sample(self.grid, xyz, mode="bilinear", align_corners=True)
        out = out.reshape(self.channels, -1).T.reshape(*shape, self.channels)
        return out


class SemanticDenseVoxel(SemanticNeRF):
    def __init__(self, args, device):
        super(SemanticDenseVoxel, self).__init__(args, device)
        self.reg = TVLoss3D()

    def init_volume(self):
        self.rgb_voxel = DenseGrid(self.rgb_dim, self.grid_size, 0).to(self.device)
        self.semantic_voxel = DenseGrid(self.semantic_dim, self.grid_size, 0).to(
            self.device
        )
        self.density_voxel = DenseGrid(self.density_dim, self.grid_size, 0).to(
            self.device
        )

    def compute_density_feature(self, xyz_sampled):
        return self.density_voxel(xyz_sampled)

    def compute_semantic_feature(self, xyz_sampled):
        return self.semantic_voxel(xyz_sampled)

    def compute_rgb_feature(self, xyz_sampled):
        return self.rgb_voxel(xyz_sampled)

    def get_kwargs(self):
        return {
            "model_type": "densevoxel",
            "rgb_dim": self.rgb_dim,
            "density_dim": self.density_dim,
            "semantic_dim": self.semantic_dim,
            "class_num": self.class_num,
            "near_far": self.near_far,
            "n_samples_max": self.n_samples_max,
            "step_ratio": self.step_ratio,
            "aabb": self.aabb.tolist(),
            "n_voxel": self.n_voxel,
            "grid_size": self.grid_size.tolist(),
            "occupancy_thres": self.occupancy_thres,
            "mesh_resolution": self.mesh_resolution,
            "level_set": self.level_set,
            "shadingmode_rgb": self.shadingmode_rgb,
            "shadingmode_density": self.shadingmode_density,
            "shadingmode_semantic": self.shadingmode_semantic,
            "pos_pe": self.pos_pe,
            "view_pe": self.view_pe,
            "fea_pe": self.fea_pe,
            "feature_rgb": self.feature_rgb,
            "feature_semantic": self.feature_semantic,
            "use_xyz": self.use_xyz,
            "use_view": self.use_view,
            "density_shift": self.density_shift,
            "raymarch_weight_thres": self.raymarch_weight_thres,
            "fea2dense_act": self.fea2dense_act,
            "use_gradientscaler": self.use_gradientscaler,
            "sampling_random_level": self.sampling_random_level,
            "radius": self.radius,
            "max_env": self.max_env,
        }

    @torch.no_grad()
    def up_sampling(self, voxel, res_target):
        voxel.grid = torch.nn.Parameter(
            F.interpolate(
                voxel.grid.data,
                size=tuple(res_target),
                mode="trilinear",
                align_corners=True,
            )
        )
        return voxel

    @torch.no_grad()
    def upsample_volume(self, n_voxel):
        """
        keep aabb the same, but upsample voxel grid resolution
        """
        print("upsampling volume!!!!!")
        self.update_grid_config(self.aabb, n_voxel)

        self.density_voxel = self.up_sampling(self.density_voxel, self.grid_size)
        self.semantic_voxel = self.up_sampling(self.semantic_voxel, self.grid_size)
        self.rgb_voxel = self.up_sampling(self.rgb_voxel, self.grid_size)

    @torch.no_grad()
    def shrink_volume(self, new_aabb=None):
        """
        shrink aabb and remove voxel out of new aabb
        """
        new_aabb = self.extract_bbox() if new_aabb is None else new_aabb
        print("shrinking volume to", new_aabb)
        new_aabb = torch.tensor(new_aabb).to(self.device)

        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (
            xyz_max - self.aabb[0]
        ) / self.units
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.grid_size]).amin(0)

        # print(t_l, b_r)
        self.density_voxel.grid = torch.nn.Parameter(
            self.density_voxel.grid.data[
                :, :, t_l[0] : b_r[0], t_l[1] : b_r[1], t_l[2] : b_r[2]
            ]
        )
        self.rgb_voxel.grid = torch.nn.Parameter(
            self.rgb_voxel.grid.data[
                :, :, t_l[0] : b_r[0], t_l[1] : b_r[1], t_l[2] : b_r[2]
            ]
        )
        self.semantic_voxel.grid = torch.nn.Parameter(
            self.semantic_voxel.grid.data[
                :, :, t_l[0] : b_r[0], t_l[1] : b_r[1], t_l[2] : b_r[2]
            ]
        )

        new_grid_size = b_r - t_l
        self.update_grid_config(new_aabb, -1, new_grid_size)

    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
        grad_vars = [
            {"params": self.density_voxel.parameters(), "lr": lr_init_spatialxyz},
            {"params": self.rgb_voxel.parameters(), "lr": lr_init_spatialxyz},
            {"params": self.semantic_voxel.parameters(), "lr": lr_init_spatialxyz},
        ]
        param_types = ["spatial", "spatial", "spatial"]

        if isinstance(self.rendermodule_rgb, torch.nn.Module):
            grad_vars += [
                {
                    "params": self.rendermodule_rgb.parameters(),
                    "lr": lr_init_network,
                }
            ]

            param_types.append("mlp")

        if isinstance(self.rendermodule_density, torch.nn.Module):
            grad_vars += [
                {
                    "params": self.rendermodule_density.parameters(),
                    "lr": lr_init_network,
                }
            ]
            param_types.append("mlp")

        if isinstance(self.rendermodule_semantic, torch.nn.Module):
            grad_vars += [
                {
                    "params": self.rendermodule_semantic.parameters(),
                    "lr": lr_init_network,
                }
            ]
            param_types.append("mlp")

        return grad_vars, param_types

    def tv_loss_density(self):
        total = self.reg(self.density_voxel.grid)
        return total

    def tv_loss_semantic(self):
        total = self.reg(self.semantic_voxel.grid)
        return total

    def tv_loss_rgb(self):
        total = self.reg(self.rgb_voxel.grid)
        return total
