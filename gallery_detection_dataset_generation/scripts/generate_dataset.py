import trimesh
import numpy as np
from tqdm import tqdm
import os
import json
import shutil
from scipy.spatial.transform import Rotation
import scipy.stats as stats
import pyvista as pv
from pyvista.plotting.plotter import Plotter
import matplotlib.pyplot as plt

PARAMS = {}
DEBUG = False


###################################################################################################################################
###################################################################################################################################
# GEOMETRIC FUNCTIONS
###################################################################################################################################
###################################################################################################################################
def T_to_pEuler(T):
    p = T[:3, 3]
    euler = Rotation.from_matrix(T[:3, :3]).as_euler("xyz")
    return p, euler


def T_to_pR(T):
    return T[:3, 3], T[:3, :3]


def T_to_R(T):
    return T[:3, :3]


def pR_to_T(p, R):
    R = np.array(R)
    p = np.array(p)
    a = np.hstack((R, np.reshape(p, (3, 1))))
    b = np.array((0, 0, 0, 1))
    return np.vstack((a, b))


def v_to_th_chi(v):
    v = np.array(v)
    vx, vy, vz = v
    chi = np.arctan2(vy, vx)
    th = np.arctan2(vz, np.linalg.norm(np.array((vx, vy)), 2))
    return th, chi


def v_to_R(v):
    v = np.array(v)
    theta, chi = v_to_th_chi(v)
    apitch = -theta
    ayaw = chi
    R = Rotation.from_euler("xyz", np.array((0, apitch, ayaw))).as_matrix()
    return R


def pv_to_T(p, v):
    p = np.array(p)
    v = np.array(v)
    R = v_to_R(v)
    return pR_to_T(p, R)


def rotate_vector(vector, euler_angles):
    vector = np.array(vector).reshape((3, -1))
    euler_angles = np.array(euler_angles)
    # Create rotation matrices for each axis
    R = Rotation.from_euler("xyz", euler_angles).as_matrix()
    # Perform the rotations
    rotated_vector = R @ vector
    return rotated_vector


###################################################################################################################################
###################################################################################################################################


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
    def __init__(self, axis_points, normalized, res=1):
        self.res = res
        self.grid = dict()
        self.normalized = normalized
        self.gaussian_width = 20
        mu = 0
        variance = 1
        sigma = np.sqrt(variance)
        x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, self.gaussian_width * 2 + 1)
        self.gaussian = stats.norm.pdf(x, mu, sigma)
        if self.normalized:
            self.gaussian /= np.max(self.gaussian)  # Normalize the gaussian
        for p in axis_points[:, :3]:
            self.add_point(p)

    def add_point(self, point: np.ndarray):
        i, j, k = np.floor(point / self.res).astype(int)
        try:
            self.grid[(i, j, k)].append(point)
        except:
            self.grid[(i, j, k)] = [point]

    def get_relevant_points(self, p, r):
        n = int(np.ceil(r / self.res) + 1)
        i_, j_, k_ = np.floor(p / self.res).astype(int)
        rlv_pts = np.zeros((0, 3))
        for i in np.arange(i_ - n, i_ + n):
            for j in np.arange(j_ - n, j_ + n):
                for k in np.arange(k_ - n, k_ + n):
                    if (i, j, k) in self.grid.keys():
                        rlv_pts = np.vstack((rlv_pts, self.grid[(i, j, k)]))
        return rlv_pts

    def get_label(self, pose, Rmat, rad):
        rlv_pts = self.get_relevant_points(pose, rad)
        Tmat = np.vstack((np.hstack((Rmat, np.reshape(pose, (3, 1)))), np.array((0, 0, 0, 1))))
        dist = np.linalg.norm(rlv_pts[:, :] - pose[:], 2, 1)
        intra_aps_dist = 0.5
        candidate_points = rlv_pts[np.abs(dist - rad) < intra_aps_dist, :]
        for n_candidate, candidate in enumerate(candidate_points):
            candidate = candidate.reshape((1, 3))
            if n_candidate == 0:
                lbl_pts = candidate
            else:
                if np.any(np.linalg.norm(candidate - lbl_pts, axis=1) < intra_aps_dist * 2.1):
                    pass
                else:
                    lbl_pts = np.vstack((lbl_pts, candidate))
        Tmatinv = np.linalg.inv(Tmat)
        if DEBUG:
            plotter = plotter_holder.data
            plotter.add_mesh(pv.PolyData(lbl_pts))
            plotter.add_axes_at_origin()
        yaws = []
        lbl_pts_1 = np.hstack([lbl_pts, np.ones((len(lbl_pts), 1))])
        for lbl_pt in lbl_pts_1:
            lbl_pt_transformed = Tmatinv @ lbl_pt.T
            if DEBUG:
                plotter.add_mesh(
                    pv.PolyData(lbl_pt_transformed[:3]),
                    color="black",
                    render_points_as_spheres=True,
                    point_size=10,
                )
            yaws.append(np.arctan2(lbl_pt_transformed[1], lbl_pt_transformed[0]))
        return self.gen_label_vector(yaws), lbl_pts

    def gen_label_vector(self, yaws):
        label_vector = np.zeros(360)
        for yaw in yaws:
            yaw_deg = np.rad2deg(yaw)
            central_idx = int(yaw_deg)
            for n in range(int(self.gaussian_width * 2 + 1)):
                label_idx = central_idx - self.gaussian_width + n
                label_vector[label_idx] = max(self.gaussian[n], label_vector[label_idx])
        return label_vector


def add_axis_at_point(plotter, p, o, o_type="euler"):
    if o_type == "euler":
        R = Rotation.from_euler("xyz", o, degrees=False).as_matrix()
    elif o_type == "matrix":
        R = o
    x_arrow = R @ np.array((1, 0, 0))
    y_arrow = R @ np.array((0, 1, 0))
    z_arrow = R @ np.array((0, 0, 1))
    plotter.add_arrows(p, x_arrow, color="r")
    plotter.add_arrows(p, y_arrow, color="g")
    plotter.add_arrows(p, z_arrow, color="b")


def add_axis_at_T(plotter, T):
    p, R = T_to_pR(T)
    xv = R @ np.array((1, 0, 0)).reshape(-1)
    yv = R @ np.array((0, 1, 0)).reshape(-1)
    zv = R @ np.array((0, 0, 1)).reshape(-1)
    p = p.reshape(-1)
    plotter.add_arrows(p, xv, color="r")
    plotter.add_arrows(p, yv, color="g")
    plotter.add_arrows(p, zv, color="b")


class PlotterHolder:
    def __init__(self):
        self.data: Plotter = None


plotter_holder = PlotterHolder()


def plot_everything(
    pv_mesh,
    aps,
    avs,
    p,
    o,
    rays,
    points,
    label_points,
    label,
    image,
):

    ###############
    # Plot the image and label, and save the image
    ###############
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, projection="polar")
    ax1.imshow(np.roll(image, 360, 1)[:, ::-1])
    ax2.plot(np.linspace(0, 2 * np.pi, 360), label)
    ax2.set_theta_zero_location("N")
    fig.savefig("/home/lorenzo/fig.png")
    ###############
    # Show the 3D representation of all the relevant info to get the label
    ###############
    plotter = plotter_holder.data
    plotter.add_axes()
    # Add mesh
    plotter.add_mesh(pv_mesh, style="wireframe")
    # Add axis points
    plotter.add_mesh(pv.PolyData(aps), color="k", render_points_as_spheres=True, point_size=5)
    plotter.add_arrows(aps, avs, mag=0.5, color="gray")
    # Add datapoint pose
    plotter.add_mesh(pv.PolyData(p), color="g", render_points_as_spheres=True, point_size=5)
    add_axis_at_point(plotter, p, o, o_type="euler")
    # Add sampled points for the image:
    plotter.add_mesh(pv.PolyData(points), color="y", render_points_as_spheres=True, point_size=3)
    # Add selected points for the label
    plotter.add_mesh(
        pv.PolyData(label_points), color="b", point_size=100, render_points_as_spheres=True
    )
    camera_pose = (np.array(p) + np.array((10, 10, 3))).tolist()
    plotter.camera_position = (camera_pose, p, (0, 0, 1))
    # Print info
    plotter.show()


def load_files_from_world_data(world_data: dict):
    mesh = trimesh.load(world_data["path_to_mesh"], force="mesh")
    pv_mesh = pv.read(world_data["path_to_mesh"])
    axis_data = np.loadtxt(world_data["path_to_axis"])
    return mesh, pv_mesh, axis_data


def setup_data_folder(path_to_dataset, world_data):
    save_folder_path = os.path.join(path_to_dataset, world_data["images_folder"])
    if os.path.isdir(save_folder_path):
        shutil.rmtree(save_folder_path)
    os.makedirs(save_folder_path)
    return save_folder_path


class DataSampleCreator:
    def __init__(self, base_rays, mesh, axis_data):
        self.ray_caster = RayCaster(base_rays=base_rays, mesh=mesh)
        self.label_generator = LabelGenerator(axis_data, res=4)

    def gen_sample(self, T):
        pass


def gen_random_pose(axis_data, fta_dist, hrange, vrange, rrange, prange):
    idx = np.random.randint(0, len(axis_data))
    x, y, z, vx, vy, vz, r, f, tid = axis_data[idx]
    a_T = pv_to_T((x, y, z), (vx, vy, vz))  # Get T of axis Point
    hdisp = r * np.random.uniform(-1, 1) * hrange
    vdisp = np.random.uniform(vrange[0], vrange[1]) + fta_dist
    r_p = (0, hdisp, vdisp)  # Relative position
    r_roll = np.random.uniform(-rrange, rrange)
    r_pitch = np.random.uniform(-prange, prange)
    r_yaw = np.random.uniform(0, 2 * np.pi)
    r_R = Rotation.from_euler("xyz", (r_roll, r_pitch, r_yaw)).as_matrix()  # Relative Rotation
    r_T = pR_to_T(r_p, r_R)
    if DEBUG:
        plotter = plotter_holder.data
        add_axis_at_T(plotter, a_T)
    return a_T @ r_T


def capture_dataset(index):
    image_width = index["info"]["image_width"]
    base_rays = gen_base_velodyne_rays(image_width)
    vrange = index["info"]["vrange"]
    hrange = index["info"]["hrange"]
    rrange = index["info"]["rrange"]
    prange = index["info"]["prange"]
    label_r = index["info"]["label_r"]
    normalized_img = index["info"]["normalized_img"]
    normalized_lbl = index["info"]["normalized_lbl"]
    max_range = index["info"]["max_range"]
    poses_dict = dict()
    total_poses = sum([index["data"][k]["n_datapoints"] for k in index["data"].keys()])
    with tqdm(total=total_poses, desc="Collecting dataset") as pbar:
        for world_name in index["data"].keys():
            # Get params
            path_to_dataset = index["info"]["path_to_dataset"]
            world_data = index["data"][world_name]
            n_datapoints = world_data["n_datapoints"]
            fta_dist = world_data["fta_dist"]
            # Setup folder
            save_folder_path = setup_data_folder(path_to_dataset, world_data)
            # Load files
            mesh, pv_mesh, axis_data = load_files_from_world_data(world_data)
            # Create ray caster and label generator
            # sample_generator = DataSampleCreator(base_rays, mesh, axis_data)
            ray_caster = RayCaster(base_rays, mesh)
            label_generator = LabelGenerator(axis_data, normalized=normalized_lbl)
            # Start capture
            poses_array = np.zeros((n_datapoints, 6))  # To save poses for data-checking purposes
            for n_pose in range(n_datapoints):
                plotter_holder.data = Plotter()
                pT = gen_random_pose(axis_data, fta_dist, hrange, vrange, rrange, prange)
                position, orientation = T_to_pEuler(pT)
                poses_array[n_pose, :] = np.hstack((position, orientation))
                depth_formatted, points, vectors, origins = ray_caster.cast_ray(
                    orientation, position
                )
                depth_formatted[depth_formatted > max_range] = 0
                image = np.reshape(depth_formatted, (16, image_width))
                if normalized_img:
                    image /= max_range
                label, label_points = label_generator.get_label(position, T_to_R(pT), label_r)
                file_name = f"{n_pose:010d}.npz"
                path_to_file = os.path.join(save_folder_path, file_name)
                with open(path_to_file, "wb+") as f:
                    np.savez(f, image=image, label=label)
                if DEBUG:
                    plot_everything(
                        pv_mesh,
                        aps=axis_data[:, :3],
                        avs=axis_data[:, 3:6],
                        p=position,
                        o=orientation,
                        rays=vectors,
                        points=points,
                        label=label,
                        label_points=label_points,
                        image=image,
                    )
                    print(
                        f"Img data:: max: {np.max(image)}, min: {np.min(image)}, avg: {np.average(image)}"
                        f"Label data:: max: {np.max(label)}, min: {np.min(label)}, avg: {np.average(label)}"
                    )
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
    linear_density,
    path_to_dataset_folder,
    folders_of_worlds,
    image_width,
    vrange,
    hrange,
    rrange,
    prange,
    max_range,
    normalized_img,
    normalized_lbl,
    label_r,
):
    index = dict()
    index["info"] = {
        "linear_density": linear_density,
        "path_to_dataset": path_to_dataset_folder,
        "image_width": image_width,
        "vrange": vrange,  # In meters!
        "hrange": hrange,  # Fraction of tunnel R!!
        "rrange": rrange,  # In rads!
        "prange": prange,  # In rads!
        "max_range": max_range,
        "normalized_img": normalized_img,
        "normalized_lbl": normalized_lbl,
        "label_r": label_r,
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
    worlds_folder = "/home/lorenzo/gazebo_worlds/training_worlds"
    index = gen_index(
        5,
        "/media/lorenzo/SAM500/datasets/gallery_detection_smooth_straight",
        [os.path.join(worlds_folder, a) for a in os.listdir(worlds_folder)],
        720,
        vrange=(0.1, 2),
        hrange=0.7,
        rrange=np.deg2rad(5),
        prange=np.deg2rad(5),
        max_range=50,
        normalized_img=True,
        normalized_lbl=True,
        label_r=6,
    )
    capture_dataset(index)


if __name__ == "__main__":
    main()
