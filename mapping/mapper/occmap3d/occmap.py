import time

import numpy as np
import torch

import torch.nn.functional as F
from tools.mesh import MeshExtractor
import open3d as o3d
from open3d.visualization import rendering

# import kaolin as kal
from tqdm import tqdm

from .utils import world_to_voxel_coordinates, bresenham_3d, plot_voxels
from .utils import prob2logodds, logodds2prob


class OccpancyGridMap3D:
    def __init__(
        self,
        map_size=[3, 3, 3],
        voxel_size=3 / 128,
        num_semantic_classes=7,
        world_origin=[1.5, 1.5, 1.5],
    ):
        # Check for GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            print("[!] No GPU detected. Defaulting to CPU.")

        self.map_size = torch.tensor(map_size).to(self.device)
        self.voxel_size = voxel_size
        self.map_dim = torch.floor(self.map_size / voxel_size).long()
        self.world_origin = torch.tensor(world_origin).float().to(self.device)

        # Create a mesh grid for voxel coordinates
        xv, yv, zv = torch.meshgrid(
            torch.arange(0, self.map_dim[0]),
            torch.arange(0, self.map_dim[1]),
            torch.arange(0, self.map_dim[2]),
            indexing="ij",
        )
        self.voxel_coords = torch.stack(
            [xv.flatten(), yv.flatten(), zv.flatten()], dim=1
        ).to(self.device)

        # Convert voxel coordinates to world coordinates
        self.world_coords = (
            self.voxel_coords * self.voxel_size + self.world_origin - self.map_size
        )

        # Initialize voxel properties as 4D Tensors
        self.voxel_weights = torch.zeros(*self.map_dim, 1).to(self.device)
        self.voxel_occupancy = torch.full((*self.map_dim, 1), 0.5).to(self.device)
        # RGB value is [0, 1]
        self.voxel_colors = torch.zeros(*self.map_dim, 3).to(self.device)
        self.voxel_semantics = torch.zeros(*self.map_dim, num_semantic_classes).to(
            self.device
        )
        self.num_semantic_classes = num_semantic_classes

        self.prob_occ = torch.Tensor([0.9]).to(self.device)
        self.prob_free = torch.Tensor([0.1]).to(self.device)
        self.prior = torch.Tensor([0.5]).to(self.device)

        self.aabb = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]]).to(self.device)
        self.invaabb_size = 2.0 / (self.aabb[1] - self.aabb[0])

    def update(self, rays, rgbs, depths, semantics, max_depth=5.0):
        """
        Args:
            rays: (H*W, 6)
            rgbs: (H*W, 3)
            depths: (H*W, 1)
            semantics: (H*W, num_semantic_classes)
            max_depth: float

        Returns:
            None
        """
        # Step1: filter invalid rays according to depth
        valid_depth_mask = (depths > 0) & (depths < max_depth)
        valid_rays = rays[valid_depth_mask.squeeze()]
        valid_rgbs = rgbs[valid_depth_mask.squeeze()]
        valid_depths = depths[valid_depth_mask]
        valid_semantics = semantics[valid_depth_mask.squeeze()]

        # Extracting ray origins and directions
        ray_origins = valid_rays[:, :3]
        ray_directions = valid_rays[:, 3:]

        # Calculating end points of the rays
        ray_end_points = ray_origins + ray_directions * valid_depths.unsqueeze(1)

        # Step2: find all voxels that are intersected by rays
        origin_points = world_to_voxel_coordinates(
            ray_origins, self.voxel_size, self.world_origin
        )
        end_points = world_to_voxel_coordinates(
            ray_end_points, self.voxel_size, self.world_origin
        )
        self.voxel_occupancy = prob2logodds(self.voxel_occupancy)

        t = time.time()
        for origin_point, end_point, rgb, semantic in tqdm(
            zip(origin_points, end_points, valid_rgbs, valid_semantics),
            desc="Processing rays",
            total=origin_points.shape[0],
        ):
            voxels = bresenham_3d(origin_point, end_point)

            # Check if each voxel is the endpoint
            valid_voxels_mask = self.is_within_bounds_vectorized(voxels)
            voxels = voxels[valid_voxels_mask]

            # Step3: update occupancy probability of each voxel according the depth
            # Accelerate: we assume last voxel is endpoint, others are free voxel
            if voxels.shape[0] > 0:
                m_prob = torch.ones(voxels.shape[0]).to(self.device) * self.prob_free
                m_prob[-1] = self.prob_occ
                m_prob = prob2logodds(m_prob) - prob2logodds(self.prior)
                self.voxel_occupancy[
                    voxels[:, 0], voxels[:, 1], voxels[:, 2]
                ] += m_prob.unsqueeze(-1)

                w = self.voxel_weights[voxels[:, 0], voxels[:, 1], voxels[:, 2]]
                wp = w + 1

                # Step4: update RGB value of each voxel according to the RGBs
                ray_rgbs = torch.zeros((w.shape[0], 3), device=self.device)
                ray_rgbs[-1] = rgb
                self.voxel_colors[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = (
                    self.voxel_colors[voxels[:, 0], voxels[:, 1], voxels[:, 2]] * w
                    + ray_rgbs
                ) / wp

                # Step5: Update semantics of each voxel according to the semantics
                ray_semantics = torch.zeros(
                    (w.shape[0], self.num_semantic_classes), device=self.device
                )
                ray_semantics[-1] = semantic
                self.voxel_semantics[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = (
                    self.voxel_semantics[voxels[:, 0], voxels[:, 1], voxels[:, 2]] * w
                    + ray_semantics
                ) / wp

                # update weights (hit number)
                self.voxel_weights[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = wp

        self.voxel_occupancy = logodds2prob(self.voxel_occupancy)

        print(f"Time taken: {time.time() - t}")

    def render(self, intrinsic, extrinsic, width=400, height=400, target_class_id=None):
        """
        Render the scene from the mesh according to intrinsic and extrinsic.

        Args:
            intrinsic: (3, 3)
            extrinsic: (4, 4)
            width: int
            height: int
            target_class_id: int, optional, default=None

        Returns:
            image: (H, W, 3) array representing the rendered image.
        """
        mesh = self.extract_mesh(target_class_id)
        render = rendering.OffscreenRenderer(width, height)
        render.scene.set_background([1.0, 1.0, 1.0, 1.0])
        render.scene.view.set_post_processing(False)

        # material for TriangleMesh
        mtl = rendering.MaterialRecord()
        mtl.shader = "defaultUnlit"

        # add geometry
        render.scene.clear_geometry()
        render.scene.add_geometry("mesh", mesh, mtl)

        # setup camera
        o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=width, height=height, intrinsic_matrix=intrinsic
        )
        render.setup_camera(intrinsics=o3d_intrinsic, extrinsic_matrix=extrinsic)

        # rgb image
        rgb_img = render.render_to_image()
        rgb_img = np.asarray(rgb_img, dtype=np.float32) / 255.0

        return rgb_img

    def render_raycasting(self, rays, require_class=None, max_distance=5.0):
        """
        Render the scene using ray-casting.

        Args:
            rays: (H*W, 6) tensor of ray origins and directions.
            max_distance: Maximum distance to trace each ray.

        Returns:
            render_image: (H*W, 3) array representing the rendered image.
        """
        rays = rays.to(self.device)
        render_image = torch.ones((rays.shape[0], 3), device=self.device)
        epsilon = 1e-10

        # Extracting ray origins and directions
        ray_origins = rays[:, :3]
        ray_directions = rays[:, 3:]

        # Calculating end points of the rays
        ray_end_points = ray_origins + ray_directions * max_distance

        # Step2: find all voxels that are intersected by rays
        origin_points = world_to_voxel_coordinates(
            ray_origins, self.voxel_size, self.world_origin
        )
        end_points = world_to_voxel_coordinates(
            ray_end_points, self.voxel_size, self.world_origin
        )

        for ray_id, (origin_point, end_point) in tqdm(
            enumerate(zip(origin_points, end_points)),
            desc="Processing rays",
            total=origin_points.shape[0],
        ):
            voxels = bresenham_3d(origin_point, end_point)
            valid_voxels_mask = self.is_within_bounds_vectorized(voxels)
            voxels = voxels[valid_voxels_mask]

            prob_occ = self.voxel_occupancy[voxels[:, 0], voxels[:, 1], voxels[:, 2]]
            semantic_prob = self.voxel_semantics[
                voxels[:, 0], voxels[:, 1], voxels[:, 2]
            ]
            semanic_label = torch.argmax(semantic_prob, dim=-1, keepdim=True)

            if require_class is not None:
                semantic_mask = semanic_label == require_class
                semantic_mask = semantic_mask.long()
                prob_occ = prob_occ.clone()
                prob_occ = prob_occ * semantic_mask

            if torch.any(prob_occ > self.prior):
                prob_free = 1 - prob_occ
                prob_free_shift = torch.cat(
                    [torch.ones_like(prob_free[:1]), prob_free], dim=0
                )
                prob_free_cum = torch.cumprod(prob_free_shift, dim=0)[:-1]
                weights = prob_free_cum * prob_occ
                weights = weights / (torch.sum(weights, dim=0) + epsilon)
                rgbs = self.voxel_colors[voxels[:, 0], voxels[:, 1], voxels[:, 2]]
                rgb = torch.sum(weights * rgbs, dim=0)

                render_image[ray_id] = rgb

        return render_image

    def search_voxels_along_ray(self, ray, max_distance=5.0):
        """
        Render the scene using ray-casting.

        Args:
            ray: (6, ) tensor of ray origins and directions.
            max_distance: Maximum distance to trace each ray.

        Returns:
            voxels: (N, 3) array representing the voxels along the ray.
        """
        voxels = []

        ray_origin = ray[:3]
        ray_direction = ray[3:]
        ray_end_point = ray_origin + ray_direction * max_distance

        origin_point = world_to_voxel_coordinates(
            ray_origin, self.voxel_size, self.world_origin
        )
        end_point = world_to_voxel_coordinates(
            ray_end_point, self.voxel_size, self.world_origin
        )

        voxels = bresenham_3d(origin_point, end_point)

        # Check if each voxel is the endpoint
        valid_voxels_mask = self.is_within_bounds_vectorized(voxels)
        voxels = voxels[valid_voxels_mask]

        return voxels

    def search_voxels_along_ray_parallel(self, rays):
        """
        Render the scene using ray-casting.

        Args:
            rays: (B, 6) tensor of ray origins and directions.

        Returns:
            voxels: list: store voxels along each ray.
        """
        new_dim = self.map_dim[0] / 2

        # convert voxel coordinates to range [-1, 1]
        shift_points = (self.voxel_coords / new_dim) - 1
        space_carving_level = torch.log2(self.map_dim[0])

        spc = kal.ops.conversions.unbatched_pointcloud_to_spc(
            shift_points, space_carving_level
        )
        octree, point_hierarchy, pyramid, prefix = (
            spc.octrees,
            spc.point_hierarchies,
            spc.pyramids[0],
            spc.exsum,
        )

        ray_origins = rays[:, :3]
        ray_directions = rays[:, 3:]
        ray_end_points = ray_origins + ray_directions

        ray_origins = world_to_voxel_coordinates(
            ray_origins, self.voxel_size, self.world_origin
        )
        ray_end_points = world_to_voxel_coordinates(
            ray_end_points, self.voxel_size, self.world_origin
        )
        ray_origins = (ray_origins / new_dim) - 1
        ray_end_points = (ray_end_points / new_dim) - 1
        ray_directions = (ray_end_points - ray_origins) / torch.norm(
            ray_end_points - ray_origins, dim=1
        ).unsqueeze(-1)

        ray_origins = ray_origins - ray_directions * 3

        nugs_ridx, nugs_pidx, depth = kal.render.spc.unbatched_raytrace(
            octree,
            point_hierarchy,
            pyramid,
            prefix,
            ray_origins,
            ray_directions,
            space_carving_level,
            with_exit=True,
        )

        nugs_ridx = nugs_ridx.long()
        origins = ray_origins[nugs_ridx]
        directions = ray_directions[nugs_ridx]
        z = (depth[:, 1] + depth[:, 0]) / 2
        voxels = origins + directions * z.reshape(-1, 1)
        voxels = torch.floor((voxels + 1) * new_dim)
        voxels = torch.clamp(voxels, 0, self.map_dim[0] - 1)

        ray_indexes, counts = torch.unique(nugs_ridx, return_counts=True)
        all_voxels = torch.split(voxels, counts.tolist())

        return all_voxels

    def extract_point_clouds(self, threshold=0.9):
        """
        Args:
            threshold: Probability threshold to determine occupancy.
        Returns:
            pcds: Numpy Array, shape: [N, 3]
        """
        occupied_voxels = self.voxel_occupancy > threshold

        # Get the indices of occupied voxels
        voxel_indices = torch.nonzero(occupied_voxels)[:, :3]

        # Convert voxel indices to world coordinates
        pcds = voxel_indices * self.voxel_size + self.world_origin - self.map_size

        # Retrieve RGB values for these voxels
        colors = self.voxel_colors[
            voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]
        ]

        # Retrieve semantic labels for these voxels
        # Using argmax to select the class with the highest probability
        semantics = self.voxel_semantics[
            voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]
        ].argmax(dim=1)

        pcds = pcds.cpu().numpy()
        colors = colors.cpu().numpy()
        semantics = semantics.cpu().numpy()

        return pcds, colors, semantics

    def is_within_bounds(self, voxel):
        """
        Check if the voxel index is within the bounds of the voxel occupancy grid.

        Args:
        - voxel: A tensor or list representing the voxel index [x, y, z].

        Returns:
        - bool: True if the voxel is within bounds, False otherwise.
        """
        x, y, z = voxel
        return (
            0 <= x < self.voxel_occupancy.shape[0]
            and 0 <= y < self.voxel_occupancy.shape[1]
            and 0 <= z < self.voxel_occupancy.shape[2]
        )

    def is_within_bounds_vectorized(self, voxels):
        """
        Check if voxels are within the bounds of the voxel grid.
        Args:
        - voxels: PyTorch tensor of shape [N, 3] representing voxel indices.

        Returns:
        - Boolean tensor of shape [N] where True indicates the voxel is within bounds.
        """
        x_within = (voxels[:, 0] >= 0) & (voxels[:, 0] < self.voxel_occupancy.shape[0])
        y_within = (voxels[:, 1] >= 0) & (voxels[:, 1] < self.voxel_occupancy.shape[1])
        z_within = (voxels[:, 2] >= 0) & (voxels[:, 2] < self.voxel_occupancy.shape[2])
        return x_within & y_within & z_within

    def save(self, filename):
        """
        Save the occupancy grid to a file.

        Args:
        - filename (str): The path to the file where the grid will be saved.
        """
        state = {
            "voxel_occupancy": self.voxel_occupancy,
            "voxel_colors": self.voxel_colors,
            "voxel_weights": self.voxel_weights,
            "voxel_semantics": self.voxel_semantics,
        }
        torch.save(state, filename)
        print(f"3D Occupancy grid saved to {filename}")

    def load(self, filename):
        """
        Load the occupancy grid from a file.

        Args:
        - filename (str): The path to the file from which the grid will be loaded.
        """
        state = torch.load(filename, map_location=self.device)

        self.voxel_occupancy = state["voxel_occupancy"]
        self.voxel_colors = state["voxel_colors"]
        self.voxel_weights = state["voxel_weights"]
        self.voxel_semantics = state["voxel_semantics"]

        print(f"3D Occupancy grid loaded from {filename}")

    def get_region_of_interest(self, target_class_id):
        voxel_labels = torch.argmax(self.voxel_semantics, dim=-1)
        # roi_mask = voxel_labels == target_class_id
        roi_mask = sum(voxel_labels == i for i in target_class_id)
        roi_index = (roi_mask).nonzero(as_tuple=False)
        return roi_index

    def save_mesh(self, path, target_class_id=[]):
        extractor = MeshExtractor(
            self,
            device=self.device,
            resolution0=32,
            target_class_id=target_class_id,
            # threshold=0.51,
        )
        mesh = extractor.generate_mesh(aabb=self.aabb)
        mesh.export(path)

    def query_occupancy(self, point, target_class_id):
        xyz = (point - self.aabb[0]) * self.invaabb_size - 1
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)

        occupancy = F.grid_sample(
            self.voxel_occupancy.permute(2, 1, 0, 3).view(1, 1, *self.map_dim),
            xyz,
            mode="bilinear",
            align_corners=True,
        )  # (1, 1, D, H, W)
        occupancy = occupancy.reshape(1, -1).T.reshape(*shape)
        occupancy = occupancy.view(-1)
        # occupancy[occupancy > 0.5] = 1.0

        if len(target_class_id) > 0:
            label_voxel = torch.argmax(self.voxel_semantics, dim=-1).type(torch.float32)
            semantics = F.grid_sample(
                label_voxel.permute(2, 1, 0).view(1, 1, *self.map_dim),
                xyz,
                mode="nearest",
                align_corners=True,
            )
            semantics = semantics.reshape(1, -1).T.reshape(*shape)
            semantics = semantics.view(-1)
            # semantic_mask = semantics == target_class_id
            semantic_mask = sum(semantics == i for i in target_class_id).bool()
            occupancy[~semantic_mask] = 0
        return occupancy

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


if __name__ == "__main__":
    occ3d = OccpancyGridMap3D()
