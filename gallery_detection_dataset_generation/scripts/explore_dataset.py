import numpy as np
import pyvista as pv
from pyvista.plotting.plotter import Plotter
from tqdm import tqdm
import os
import json
from scipy.spatial.transform import Rotation
import scipy.stats as stats
import matplotlib.pyplot as plt
from generate_dataset import rotate_vector

PARAMS = {"path_to_dataset_folder": "/media/lorenzo/SAM500/datasets/tested_dataset"}


def main(P):
    path_to_dataset_folder = P["path_to_dataset_folder"]
    path_to_index_file = os.path.join(path_to_dataset_folder, "index.json")
    path_to_poses_file = os.path.join(path_to_dataset_folder, "poses.json")
    with open(path_to_index_file, "r") as f:
        index = json.load(f)
    with open(path_to_poses_file, "r") as f:
        poses = json.load(f)
    for world_name in index["data"].keys():
        wdata = index["data"][world_name]
        path_to_mesh = wdata["path_to_mesh"]
        path_to_axis = wdata["path_to_axis"]
        wposes = poses[world_name]
        pv_mesh = pv.read(path_to_mesh)
        axis_data = np.loadtxt(path_to_axis)
        plotter = Plotter()
        mesh_actor = plotter.add_mesh(pv_mesh, style="wireframe")
        aps_actor = plotter.add_mesh(
            pv.PolyData(axis_data[:, :3]),
            render_points_as_spheres=True,
            point_size=10,
            color="r",
        )
        avs_actor = plotter.add_arrows(axis_data[:, :3], axis_data[:, 3:6])
        plotter.show()
        for n_pose in range(len(wposes)):
            pose = wposes[n_pose]
            data_sample_file_name = f"{n_pose:010d}.npz"
            data_sample_path = os.path.join(
                path_to_dataset_folder, world_name, data_sample_file_name
            )
            data_sample = np.load(data_sample_path)
            lbl, img = data_sample["label"], data_sample["image"]
            print(f"max(img)={np.max(img)}")
            print(f"min(img)={np.min(img)}")
            print(f"avg(img)={np.mean(img)}")
            # Plot and save sample
            fig = plt.figure()
            ax1 = fig.add_subplot(2, 1, 1)
            ax2 = fig.add_subplot(2, 1, 2, projection="polar")
            ax2.set_theta_zero_location("N")
            ax1.imshow(img)
            ax2.plot(np.linspace(0, 2 * np.pi, 360), lbl)
            plt.savefig("/home/lorenzo/fig.png")
            # Plot and show environment
            plotter = Plotter()
            center = np.array(pose[:3])
            direction = rotate_vector(np.array((1, 0, 0)), pose[3:6])
            print(center.shape)
            print(direction.shape)
            plotter.add_arrows(center, direction, color="g")
            plotter.add_actor(mesh_actor)
            plotter.add_actor(aps_actor)
            plotter.add_actor(avs_actor)
            cp = plotter.camera_position
            cp = list(cp)
            cp[0] = pose[:3]
            cp[1] = pose[:3]
            cp[1][
                1
            ] += 1  # If the focal point of the camera is the same as position nothing is shown
            cp[2] = (0, 0, 1)
            plotter.camera_position = cp
            plotter.show()
    os.remove("/home/lorenzo/fig.png")


if __name__ == "__main__":
    main(PARAMS)
