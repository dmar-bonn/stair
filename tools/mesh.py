import torch
import torch.optim as optim
from torch import autograd
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import trimesh
from . import libmcubes
from .libmise import MISE
import time
from skimage.morphology import binary_dilation, disk
import skimage.measure
import plyfile
from tqdm import trange
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix, eye


# TODO Output masking yes or no
class MeshExtractor(object):
    """Mesh extractor class for Occupancies

    The class contains functions for exctracting the meshes from a occupancy field

    Args:
        model (nn.Module): trained model
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        device (device): pytorch device
        resolution0 (int): start resolution for MISE
        upsampling steps (int): number of upsampling steps
        padding (float): how much padding should be used for MISE
    """

    def __init__(
        self,
        model,
        points_batch_size=200000,
        threshold=0.5,
        device=None,
        resolution0=64,
        upsampling_steps=3,
        padding=0.1,
        target_class_id=[],
    ):
        self.model = model
        self.points_batch_size = points_batch_size
        self.threshold = threshold
        self.device = device
        self.resolution0 = resolution0
        self.upsampling_steps = upsampling_steps
        self.padding = padding
        self.target_class_id = target_class_id

    @torch.no_grad()
    def generate_mesh(self, aabb):
        """Generates the output mesh.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        """
        # self.model.eval()
        stats_dict = {}
        # self.threshold = np.log(self.threshold) - np.log(1.0 - self.threshold)
        # self.threshold = 0.5
        t0 = time.time()
        # Compute bounding box size
        self.box_size = aabb[1] - aabb[0]  #  + self.padding
        # Shortcut
        mesh_extractor = MISE(self.resolution0, self.upsampling_steps, self.threshold)

        points = mesh_extractor.query()

        while points.shape[0] != 0:
            # Query points
            pointsf = torch.FloatTensor(points).to(self.device)
            # Normalize to bounding box
            pointsf = pointsf / mesh_extractor.resolution
            pointsf = self.box_size * (pointsf - 0.5)
            # Evaluate model and update
            values = self.eval_points(pointsf).cpu().numpy()

            values = values.astype(np.float64)
            mesh_extractor.update(points, values)
            points = mesh_extractor.query()

        value_grid = mesh_extractor.to_dense()

        # Extract mesh
        stats_dict["time (eval points)"] = time.time() - t0
        mesh = self.extract_mesh(value_grid, stats_dict=stats_dict)
        if not self.is_empty_mesh(mesh):
            mesh = trimesh.smoothing.filter_laplacian(
                mesh,
                lamb=0.5,
                iterations=5,
                implicit_time_integration=False,
                volume_constraint=False,
                laplacian_operator=None,
            )

        return mesh

    def is_empty_mesh(self, mesh):
        vertices_num = mesh.vertices.shape[0]
        faces_num = mesh.faces.shape[0]
        if vertices_num == 0 or faces_num == 0:
            return True
        else:
            return False

    @torch.no_grad()
    def eval_points(self, p):
        """Evaluates the occupancy values for the points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        """

        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []
        for pi in p_split:
            pi = pi.unsqueeze(0).to(self.device)
            with torch.no_grad():
                occ_hat = self.model.query_occupancy(pi, self.target_class_id)

            occ_hats.append(occ_hat.detach().cpu())

        occ_hat = torch.cat(occ_hats, dim=0)

        return occ_hat

    def extract_mesh(self, occ_hat, stats_dict=dict()):
        """Extracts the mesh from the predicted occupancy grid.

        Args:
            occ_hat (tensor): value grid of occupancies
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        """
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        # box_size = 2 + self.padding
        t0 = time.time()
        occ_hat_padded = np.pad(occ_hat, 1, "constant", constant_values=-1e6)

        # vertices, triangles = libmcubes.marching_cubes(occ_hat_padded, self.threshold)
        vertices, triangles = libmcubes.marching_cubes(occ_hat, self.threshold)
        stats_dict["time (marching cubes)"] = time.time() - t0
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # Undo padding
        # vertices -= 1
        # Normalize to bounding box
        vertices /= np.array([n_x - 1, n_y - 1, n_z - 1])
        # vertices /= np.array([n_x, n_y, n_z])
        vertices = self.box_size.cpu().numpy() * (vertices - 0.5)

        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles, process=False)

        return mesh
