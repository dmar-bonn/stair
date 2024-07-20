import torch
import torch.nn
import torch.nn.functional as F
from .sh import eval_sh_bases


def positional_encoding(positions, freqs):
    freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1],)
    )  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts


def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1.0 - torch.exp(-sigma * dist)
    T = torch.cumprod(
        torch.cat(
            [torch.ones(alpha.shape[0], 1).to(alpha.device), 1.0 - alpha + 1e-10], -1
        ),
        -1,
    )[:, :-1]

    weights = alpha * T  # [N_rays, N_samples]
    return alpha, weights, T


def occ2weight(occ):
    # occ  [N_rays, N_samples]
    T = torch.cumprod(
        torch.cat([torch.ones(occ.shape[0], 1).to(occ.device), 1.0 - occ], -1),
        -1,
    )[:, :-1]
    weights = occ * T  # [N_rays, N_samples]
    return weights, T


### semantic renderer ###


class SemanticCNN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, fea_dim):
        super(SemanticCNN, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(
            in_dim, fea_dim, kernel_size=(3, 3), padding=1, bias=False
        )
        self.conv2d_2 = torch.nn.Conv2d(
            fea_dim, out_dim, kernel_size=(3, 3), padding=1, bias=False
        )
        self.pool = torch.nn.AvgPool2d(3, stride=1)
        self.activation = torch.nn.ReLU(inplace=True)

    def forward(self, feature_map):
        x = self.pool(self.conv2d_1(feature_map))
        x = self.activation(x)
        x = self.activation(self.conv2d_2(x))
        return x


def SemanticRender(pts, features):
    semantic = features
    return semantic


class SemanticRenderMLP(torch.nn.Module):
    def __init__(self, in_dim, class_num, feature_dim=128, use_xyz=False):
        super(SemanticRenderMLP, self).__init__()
        self.use_xyz = use_xyz
        if use_xyz:
            in_dim += 3

        layer1 = torch.nn.Linear(in_dim, feature_dim)
        layer2 = torch.nn.Linear(feature_dim, feature_dim)
        layer3 = torch.nn.Linear(feature_dim, class_num)
        self.mlp = torch.nn.Sequential(
            layer1,
            torch.nn.ReLU(inplace=True),
            layer2,
            torch.nn.ReLU(inplace=True),
            layer3,
        )
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, features):
        indata = [features]
        if self.use_xyz:
            indata += [pts]

        mlp_in = torch.cat(indata, dim=-1)
        semantic = self.mlp(mlp_in)
        return semantic


### rgb renderer ###


def RGBRender(pts, viewdirs, features):
    rgb = features
    return rgb


def SHRender(pts, viewdirs, features):
    sh_mult = eval_sh_bases(2, viewdirs)[:, None]
    rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
    rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
    return rgb


def DensityRender(pts, features):
    density = features
    return density


class DensityRenderMLP(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        pos_pe=6,
    ):
        super(DensityRenderMLP, self).__init__()
        self.pos_pe = pos_pe

        in_dim += 2 * pos_pe * 3

        self.layer = torch.nn.Linear(in_dim, 1)
        torch.nn.init.constant_(self.layer.bias, 0)
        torch.nn.init.constant_(self.layer.weight, 0.1)

    def forward(self, pts, features):
        indata = [features]
        if self.pos_pe > 0:
            indata += [positional_encoding(pts, self.pos_pe)]

        mlp_in = torch.cat(indata, dim=-1)
        density = self.layer(mlp_in)
        return density  # (B, N, 1)


class RGBRenderMLP(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        pos_pe=6,
        view_pe=6,
        fea_pe=6,
        feature_dim=128,
        use_xyz=False,
        use_view=False,
    ):
        super(RGBRenderMLP, self).__init__()
        self.use_xyz = use_xyz
        self.use_view = use_view
        self.pos_pe = pos_pe
        self.view_pe = view_pe
        self.fea_pe = fea_pe

        if use_xyz:
            in_dim += 3
        if use_view:
            in_dim += 3

        in_dim += self.use_view * 2 * view_pe * 3 + 2 * pos_pe * 3 + 2 * fea_pe * in_dim

        layer1 = torch.nn.Linear(in_dim, feature_dim)
        layer2 = torch.nn.Linear(feature_dim, feature_dim)
        layer3 = torch.nn.Linear(feature_dim, 3)

        self.mlp = torch.nn.Sequential(
            layer1,
            torch.nn.ReLU(inplace=True),
            layer2,
            torch.nn.ReLU(inplace=True),
            layer3,
        )
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features]

        if self.use_xyz:
            indata += [pts]
        if self.use_view:
            indata += [viewdirs]

        if self.fea_pe > 0:
            indata += [positional_encoding(features, self.fea_pe)]
        if self.pos_pe > 0:
            indata += [positional_encoding(pts, self.pos_pe)]
        if self.use_view and self.view_pe > 0:
            indata += [positional_encoding(viewdirs, self.view_pe)]

        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class OccGridMask(torch.nn.Module):
    def __init__(self, device, aabb, occ_volume):
        super(OccGridMask, self).__init__()
        self.device = device

        self.aabb = aabb.to(self.device)
        self.aabb_size = self.aabb[1] - self.aabb[0]
        self.invgrid_size = 1.0 / self.aabb_size * 2
        self.occ_volume = occ_volume.view(1, 1, *occ_volume.shape[-3:])
        self.grid_size = torch.LongTensor(
            [occ_volume.shape[-1], occ_volume.shape[-2], occ_volume.shape[-3]]
        ).to(self.device)

    def sample_occupancy(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        occ_vals = F.grid_sample(
            self.occ_volume, xyz_sampled.view(1, -1, 1, 1, 3), align_corners=True
        ).view(-1)

        return occ_vals

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled - self.aabb[0]) * self.invgrid_size - 1


class OcclusionMap:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.unknown_grid = torch.zeros([*self.grid_size])
        self.known_grid = torch.zeros([*self.grid_size])

    def update_state(self, xyz_unknown, xyz_known):
        points_unknown_index = self.coordinate2index(xyz_unknown, require_unique=True)
        points_known_index = self.coordinate2index(xyz_known, require_unique=True)
        self.unknown_grid[points_unknown_index] = 1
        self.known_grid[points_known_index] = 1

    def query_state(self, xyz):

        points_index = self.coordinate2index(xyz)
        result = (self.unknown_grid[points_index] == 1) * (
            self.known_grid[points_index] == 0
        )
        return result

    def coordinate2index(self, points_co, require_unique=False):
        points_index = torch.floor(self.grid_size * (0.999999 + points_co) / 2).type(
            torch.long
        )
        if require_unique:
            points_index = torch.unique(points_index, dim=0)

        points_index = torch.transpose(points_index, 0, 1)
        return list(points_index)


class GradientScalerOcc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, occ, ray_dist):
        # ray_dist = 0 * ray_dist
        ctx.save_for_backward(ray_dist)
        return occ, ray_dist

    @staticmethod
    def backward(
        ctx,
        grad_output_occ,
        grad_output_ray_dist,
    ):
        (ray_dist,) = ctx.saved_tensors
        scaling = torch.square(ray_dist).clamp(0, 1)
        grad_output_occ = grad_output_occ * scaling
        return (
            grad_output_occ,
            grad_output_ray_dist,
        )


class GradientScalerRGB(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rgb, ray_dist):
        # ray_dist = 0 * ray_dist
        ctx.save_for_backward(ray_dist)
        return rgb, ray_dist

    @staticmethod
    def backward(
        ctx,
        grad_output_rgb,
        grad_output_ray_dist,
    ):
        (ray_dist,) = ctx.saved_tensors
        scaling = torch.abs(ray_dist).clamp(0, 1)
        grad_output_rgb = grad_output_rgb * scaling.unsqueeze(-1)
        return (
            grad_output_rgb,
            grad_output_ray_dist,
        )
