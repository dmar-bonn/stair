from scipy.spatial.transform import Rotation as R
import numpy as np
import json


def rotation_2_quaternion(rotation_matrix):
    r = R.from_matrix(rotation_matrix)
    return r.as_quat()


def xyz_to_view(xyz, radius):
    phi = np.arcsin(xyz[2] / radius)  # phi from 0 to 0.5*pi
    theta = np.arctan2(xyz[1], xyz[0]) % (2 * np.pi)  # theta from 0 to 2*pi

    return [phi, theta]


def view_to_pose(view, radius):
    phi, theta = view

    # phi should be within [min_phi, 0.5*np.pi)
    if phi >= 0.5 * np.pi:
        phi = np.pi - phi

    pose = np.eye(4)
    x = radius * np.cos(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.cos(phi)
    z = radius * np.sin(phi)

    translation = np.array([x, y, z])
    rotation = R.from_euler("ZYZ", [theta, -phi, np.pi]).as_matrix()

    pose[:3, -1] = translation
    pose[:3, :3] = rotation
    return pose


def get_camera_view_direction(poses):
    view_direction = poses[..., :3, 0]
    view_direction = view_direction / np.linalg.norm(view_direction)
    return view_direction


def view_to_pose_batch(views, radius):
    num = len(views)
    phi = views[:, 0]
    theta = views[:, 1]

    # phi should be within [min_phi, 0.5*np.pi)
    index = phi >= 0.5 * np.pi
    phi[index] = np.pi - phi[index]

    poses = np.broadcast_to(np.identity(4), (num, 4, 4)).copy()

    x = radius * np.cos(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.cos(phi)
    z = radius * np.sin(phi)

    translations = np.stack((x, y, z), axis=-1)

    angles = np.stack((theta, -phi, np.pi * np.ones(num)), axis=-1)
    rotations = R.from_euler("ZYZ", angles).as_matrix()

    poses[:, :3, -1] = translations
    poses[:, :3, :3] = rotations

    return poses


def local_uniform_sampling(radius, xyz, view_change, phi_min):
    """
    random scatter view direction changes by given current position and view change range.
    """

    u = xyz / np.linalg.norm(xyz)
    w = np.array([0.0, 0.0, 0.0])
    view = [0, 0]

    while np.linalg.norm(w) < 0.001 or view[0] < phi_min:
        # pick a random vector:
        r = np.random.multivariate_normal(np.zeros_like(u), np.eye(len(u)))

        # form a vector perpendicular to u:
        uperp = r - r.dot(u) * u
        uperp = uperp / np.linalg.norm(uperp)

        # random view angle change in radian
        random_view_change = np.random.uniform(low=view_change[0], high=view_change[1])
        cosine = np.cos(random_view_change)
        w = cosine * u + np.sqrt(1 - cosine**2 + 1e-8) * uperp
        w = radius * w / np.linalg.norm(w)

        view = xyz_to_view(w, radius)

    return view


def uniform_sampling(radius, phi_min):
    """
    uniformly generate unit vector on hemisphere.
    then calculate corresponding view direction targeting coordinate origin.
    """

    xyz = np.array([0.0, 0.0, 0.0])
    view = [0, 0]

    # avoid numerical error and minimal height requirement
    while np.linalg.norm(xyz) < 0.001 or view[0] < phi_min:
        xyz[0] = np.random.uniform(low=-1.0, high=1.0)
        xyz[1] = np.random.uniform(low=-1.0, high=1.0)
        xyz[2] = np.random.uniform(low=0.0, high=1.0)

        xyz = radius * xyz / np.linalg.norm(xyz)
        view = xyz_to_view(xyz, radius)

    return view


def focal_len_to_fov(focal, resolution):
    """
    calculate FoV based on given focal length adn image resolution

    Args:
        focal: [fx, fy]
        resolution: [W, H]

    Returns:
        FoV: [HFoV, VFoV]

    """
    focal = np.asarray(focal)
    resolution = np.asarray(resolution)

    return 2 * np.arctan(0.5 * resolution / focal)


def gazebo2opencv(pose):
    # transform gazebo coordinate to opencv format
    transformation = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return pose @ transformation


def record_meta_data(path, camera_info, trajectory, target_class_id, save_list):
    resolution = camera_info["image_resolution"]
    c = camera_info["c"]
    focal = camera_info["focal"]

    fov = focal_len_to_fov(focal, resolution)

    record_dict = {}
    record_dict["camera_angle_x"] = fov[0]
    record_dict["camera_angle_y"] = fov[1]
    record_dict["fl_x"] = focal[0]
    record_dict["fl_y"] = focal[1]
    record_dict["k1"] = 0.0
    record_dict["k2"] = 0.0
    record_dict["p1"] = 0.0
    record_dict["p2"] = 0.0
    record_dict["cx"] = c[0]
    record_dict["cy"] = c[1]
    record_dict["h"] = resolution[0]
    record_dict["w"] = resolution[1]
    record_dict["scale"] = 1.0
    record_dict["aabb_scale"] = 2.0
    record_dict["target_class_id"] = target_class_id
    record_dict["save_list"] = save_list

    record_dict["frames"] = []

    for i, pose in enumerate(trajectory):
        image_file = f"images/{i+1:04d}.png"
        gt_semantic_file = f"semantics/gt_{i+1:04d}.npy"
        gt_semantic_map_file = f"semantics/gt_{i+1:04d}.png"
        pseudo_semantic_file = f"semantics/pseudo_{i+1:04d}.npy"
        pseudo_semantic_map_file = f"semantics/pseudo_{i+1:04d}.png"
        depth_file = f"depths/depth_{i+1:04d}.npy"

        data_frame = {
            "image_path": image_file,
            "depth_path": depth_file,
            "gt_semantic_path": gt_semantic_file,
            "gt_semantic_map_path": gt_semantic_map_file,
            "pseudo_semantic_path": pseudo_semantic_file,
            "pseudo_semantic_map_path": pseudo_semantic_map_file,
            "transform_matrix": pose.tolist(),
        }
        record_dict["frames"].append(data_frame)

    with open(f"{path}/transforms.json", "w") as f:
        json.dump(record_dict, f, indent=4)


# https://github.com/matt77hias/fibpy/blob/master/src/sampling.py
def fibonacci_spiral_hemisphere(samples_num, radius, phi_min=0, mode=0):
    n = 2 * samples_num
    rn = range(samples_num, n)

    shift = 1.0 if mode == 0 else n * np.random.random()

    ga = np.pi * (3.0 - np.sqrt(5.0))
    offset = 1.0 / samples_num

    view_samples = np.zeros((samples_num, 2))
    j = 0
    for i in rn:
        phi = ga * ((i + shift) % n)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        cos_theta = ((i + 0.5) * offset) - 1.0
        sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
        xyz = radius * np.array([cos_phi * sin_theta, sin_phi * sin_theta, cos_theta])
        view = xyz_to_view(xyz, radius)

        if view[0] < phi_min:
            view[0] = phi_min
        view_samples[j, :] = view

        j += 1
    return view_samples
