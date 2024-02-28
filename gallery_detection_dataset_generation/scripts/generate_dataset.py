import trimesh
import numpy as np
from tqdm import tqdm
import os
import json
import shutil
from scipy.spatial.transform import Rotation
import scipy.stats as stats

PARAMS = {}


def gen_base_velodyne_rays(width):
    ones = np.ones((16, width))
    theta = np.reshape(np.linspace(np.deg2rad(15), np.deg2rad(-15), 16), (16, 1))
    chi = np.reshape(np.linspace(0, 2 * np.pi, width), (1, width))
    thetas = np.reshape(ones * theta, -1)
    chis = np.reshape(ones * chi, -1)
    x = np.cos(thetas) * np.cos(chis)
    y = np.cos(thetas) * np.sin(chis)
    z = np.sin(thetas)
    vectors = np.vstack((x, y, z))
    return vectors


def rotate_vector(vector, euler_angles):
    # Create rotation matrices for each axis
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(euler_angles[0]), -np.sin(euler_angles[0])],
            [0, np.sin(euler_angles[0]), np.cos(euler_angles[0])],
        ]
    )
    Ry = np.array(
        [
            [np.cos(euler_angles[1]), 0, np.sin(euler_angles[1])],
            [0, 1, 0],
            [-np.sin(euler_angles[1]), 0, np.cos(euler_angles[1])],
        ]
    )
    Rz = np.array(
        [
            [np.cos(euler_angles[2]), -np.sin(euler_angles[2]), 0],
            [np.sin(euler_angles[2]), np.cos(euler_angles[2]), 0],
            [0, 0, 1],
        ]
    )
    # Perform the rotations
    rotated_vector = Rz @ (Ry @ (Rx @ vector))
    return rotated_vector


class RayCaster(object):
    def __init__(self, base_rays, mesh: trimesh.Trimesh):
        self.base_rays = base_rays
        self.mesh = mesh
        self.scene = self.mesh.scene()

    def cast_ray(self, cam_rot, cam_t):
        """
        :param cam_rot: (3,) array, camera rotation in Euler angle format.
        :param cam_t: (3,) array, camera translation.
        """
        vectors = rotate_vector(self.base_rays, cam_rot).T
        origins = np.ones(vectors.shape) * cam_t
        points, index_ray, index_tri = self.mesh.ray.intersects_location(
            origins, vectors, multiple_hits=False
        )
        depth_raw = np.linalg.norm(points - cam_t, 2, 1)
        depth_formatted = np.zeros(len(vectors))
        depth_formatted[index_ray] = depth_raw
        return depth_formatted, points, vectors, origins


class LabelGenerator:
    def __init__(self, axis_points, res=1):
        self.res = res
        self.grid = dict()
        self.gaussian_width = 15
        mu = 0
        variance = 1
        sigma = np.sqrt(variance)
        x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, self.gaussian_width * 2 + 1)
        self.gaussian = stats.norm.pdf(x, mu, sigma)
        for p in axis_points[:, :3]:
            self.add_point(p)

    def add_point(self, point: np.ndarray):
        i, j, k = np.floor(point / self.res).astype(int)
        try:
            self.grid[(i, j, k)].append(point)
        except:
            self.grid[(i, j, k)] = [point]

    def get_relevant_points(self, p, r):
        n = int(np.ceil(r / self.res))
        i_, j_, k_ = np.floor(p / self.res).astype(int)
        rlv_pts = np.zeros((0, 3))
        for i in np.arange(i_ - n, i_ + n):
            for j in np.arange(j_ - n, j_ + n):
                for k in np.arange(k_ - n, k_ + n):
                    if (i, j, k) in self.grid.keys():
                        rlv_pts = np.vstack((rlv_pts, self.grid[(i, j, k)]))
        return rlv_pts

    def get_label(self, point, rot, r):
        rlv_pts = self.get_relevant_points(point, r)
        dist = np.linalg.norm(rlv_pts[:, :2] - point[:2], 2, 1)
        lbl_pts = rlv_pts[np.abs(dist - r) < 0.3, :]
        vects = lbl_pts - point
        rot_inv = np.linalg.inv(rot)
        yaws = []
        for vect in vects:
            vect_t = rot_inv @ vect
            yaws.append(np.arctan2(vect_t[1], vect_t[0]))
        return self.gen_label_vector(yaws)

    def gen_label_vector(self, yaws):
        label_vector = np.zeros(360)
        for yaw in yaws:
            yaw_deg = np.rad2deg(yaw)
            central_idx = int(yaw_deg)
            for n in range(int(self.gaussian_width * 2 + 1)):
                label_idx = central_idx - self.gaussian_width - n
                label_vector[label_idx] = max(self.gaussian[n], label_vector[label_idx])
        return label_vector


def capture_dataset(index):
    image_width = index["info"]["image_width"]
    base_rays = gen_base_velodyne_rays(image_width)
    vrange = index["info"]["vrange"]
    rrange = index["info"]["rrange"]
    prange = index["info"]["prange"]
    poses_dict = dict()
    total_poses = sum([index["data"][k]["n_datapoints"] for k in index["data"].keys()])
    with tqdm(total=total_poses, desc="Collecting dataset") as pbar:
        for world_name in index["data"].keys():
            # Get params
            path_to_dataset = index["info"]["path_to_dataset"]
            n_datapoints = index["data"][world_name]["n_datapoints"]
            path_to_mesh = index["data"][world_name]["path_to_mesh"]
            path_to_axis = index["data"][world_name]["path_to_axis"]
            folder_name = index["data"][world_name]["images_folder"]
            fta_dist = index["data"][world_name]["fta_dist"]
            # Setup folder
            save_folder_path = os.path.join(path_to_dataset, folder_name)
            if os.path.isdir(save_folder_path):
                shutil.rmtree(save_folder_path)
            os.makedirs(save_folder_path)
            # Load files
            mesh = trimesh.load(path_to_mesh, force="mesh")
            axis_data = np.loadtxt(path_to_axis)
            # Create ray caster and label generator
            ray_caster = RayCaster(base_rays=base_rays, mesh=mesh)
            label_generator = LabelGenerator(axis_data, res=4)
            # Start capture
            poses_array = np.zeros((n_datapoints, 6))  # To save poses for data-checking purposes
            for n_pose in range(n_datapoints):
                idx = np.random.randint(0, len(axis_data))
                x, y, z, vx, vy, vz, r, f, tid = axis_data[idx]
                theta, chi = direction_vector_to_spherical(np.array((vx, vy, vz)))
                base_yaw = theta
                base_pitch = -chi
                perp_yaw = base_yaw + np.deg2rad(90)
                hdisp = r * np.random.uniform(-1, 1) * 0.5
                vdisp = np.random.uniform(vrange[0], vrange[1])
                roll = np.random.uniform(-rrange, rrange)
                pitch = np.random.uniform(-prange, prange)
                yaw = np.random.uniform(0, 2 * np.pi)
                hvect = np.array((np.cos(perp_yaw), np.sin(perp_yaw), 0))
                vvect = np.array((0, 0, 1))
                yaw += np.deg2rad(10)
                oR = Rotation.from_euler(
                    "xyz", np.array((0, base_pitch, base_yaw)), degrees=False
                ).as_matrix()
                nR = Rotation.from_euler("xyz", (roll, pitch, yaw), degrees=False).as_matrix()
                fR = oR @ nR
                position = np.array((x, y, z)) + (vdisp + fta_dist) * vvect + hdisp * hvect
                orientation = Rotation.from_matrix(fR).as_euler(("xyz"))
                poses_array[n_pose, :] = np.hstack((position, orientation))
                depth_formatted, points, vectors, origins = ray_caster.cast_ray(
                    orientation, position
                )
                image = np.reshape(depth_formatted, (16, image_width))
                label = label_generator.get_label(position, fR, 5)
                file_name = f"{n_pose:010d}.npz"
                path_to_file = os.path.join(save_folder_path, file_name)
                with open(path_to_file, "wb+") as f:
                    np.savez(f, image=image, label=label)
                pbar.update(1)
            poses_dict[world_name] = poses_array.tolist()
    path_to_poses_file = os.path.join(index["info"]["path_to_dataset"], "poses.json")
    with open(path_to_poses_file, "w+") as f:
        json.dump(poses_dict, f)


def spherical_to_direction_vector(theta, phi):
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    direction_vector = np.array([x, y, z])
    return direction_vector


def direction_vector_to_spherical(dv):
    theta = np.arctan2(dv[1], dv[0])
    phi = np.arcsin(dv[2])
    return theta, phi


def random_non_repeat_ints(max_int, num_ints) -> np.ndarray:
    n_repeats = int(np.floor(num_ints / max_int))
    if n_repeats == 0:
        ints = np.arange(0, max_int, dtype=int)
        np.random.shuffle(ints)
        ints = ints[:num_ints]
    else:
        ints = np.arange(0, max_int, dtype=int)
        np.random.shuffle(ints)
        ints = ints[: (num_ints - n_repeats * max_int)]
        for _ in range(n_repeats):
            ints_to_cat = np.arange(0, max_int, dtype=int)
            np.random.shuffle(ints_to_cat)
            ints = np.concatenate((ints, ints_to_cat))
    return ints


def gen_index(
    linear_density, path_to_dataset_folder, folders_of_worlds, image_width, vrange, rrange, prange
):
    index = dict()
    index["info"] = {
        "linear_density": linear_density,
        "path_to_dataset": path_to_dataset_folder,
        "image_width": image_width,
        "vrange": vrange,  # In meters!
        "rrange": rrange,  # In rads!
        "prange": prange,  # In rads!
    }
    index["data"] = dict()
    n_pts_in_dataset = 0
    for folder_of_world in folders_of_worlds:
        world_name = os.path.split(folder_of_world)[-1]
        path_to_mesh = os.path.join(folder_of_world, "mesh.obj")
        path_to_axis = os.path.join(folder_of_world, "axis.txt")
        path_to_fta = os.path.join(folder_of_world, "fta_dist.txt")
        axis_data = np.loadtxt(path_to_axis)
        fta_dist = np.loadtxt(path_to_fta)
        p1, p2 = axis_data[:2, :3]
        d = np.linalg.norm(p1 - p2)
        total_distance = d * len(axis_data)
        n_datapoints = int(total_distance * linear_density)
        n_pts_in_dataset += n_datapoints
        index["data"][world_name] = dict()
        index["data"][world_name]["path_to_mesh"] = path_to_mesh
        index["data"][world_name]["path_to_axis"] = path_to_axis
        index["data"][world_name]["total_distance"] = total_distance
        index["data"][world_name]["n_datapoints"] = n_datapoints
        index["data"][world_name]["images_folder"] = world_name
        index["data"][world_name]["fta_dist"] = fta_dist.item()
    decission = input(f"The number of datapoints is {n_pts_in_dataset}, continue? [yes]/no: ")
    if decission.lower() == "no":
        exit()
    os.makedirs(path_to_dataset_folder, exist_ok=True)
    with open(os.path.join(path_to_dataset_folder, "index.json"), "w+") as f:
        json.dump(index, f)
    return index


def main():
    worlds_folder = "/home/lorenzo/gazebo_worlds/procedural_tunnels"
    index = gen_index(
        0.005,
        "/media/lorenzo/SAM500/datasets/test_dataset",
        [os.path.join(worlds_folder, a) for a in os.listdir(worlds_folder)],
        1024,
        (0.10, 1),
        4,
        4,
    )
    capture_dataset(index)


if __name__ == "__main__":
    main()
