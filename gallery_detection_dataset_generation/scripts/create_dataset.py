import PIL.Image
import trimesh
import numpy as np
from typing import List
from functools import lru_cache

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
    Rx = np.array([[1, 0, 0], [0, np.cos(euler_angles[0]), -np.sin(euler_angles[0])], [0, np.sin(euler_angles[0]), np.cos(euler_angles[0])]])
    Ry = np.array([[np.cos(euler_angles[1]), 0, np.sin(euler_angles[1])], [0, 1, 0], [-np.sin(euler_angles[1]), 0, np.cos(euler_angles[1])]])
    Rz = np.array([[np.cos(euler_angles[2]), -np.sin(euler_angles[2]), 0], [np.sin(euler_angles[2]), np.cos(euler_angles[2]), 0], [0, 0, 1]])
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
        origins = np.reshape(np.repeat(cam_t, len(vectors), 0), vectors.shape)
        points, index_ray, index_tri = self.mesh.ray.intersects_location(origins, vectors, multiple_hits=False)
        depth_raw = np.linalg.norm(points - cam_t, 2, 1)
        depth_formatted = np.zeros(len(vectors))
        depth_formatted[index_ray] = depth_raw
        return depth_formatted


def main():
    pass


if __name__ == "__main__":
    main(PARAMS)
