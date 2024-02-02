from xml.etree.ElementTree import parse, Element
import trimesh
import os
import pathlib
import numpy as np
from trimesh import creation
import pysdf


def gazebo_models_folders():
    gazebo_model_paths = os.environ["GAZEBO_MODEL_PATH"].split(":")
    gazebo_model_paths.append("/home/lorenzo/.gazebo/models")
    gazebo_model_paths.append("/usr/share/gazebo-11/models")
    gazebo_model_paths.append("/home/lorenzo/catkin_data/downloaded_models/subt_models")
    gazebo_model_paths.append("/home/lorenzo/.ignition/fuel/fuel.gazebosim.org/openrobotics/models/cave_straight_02")
    gazebo_model_paths.append("/home/lorenzo/catkin_ws/src/somport-gazebo/models")
    gazebo_model_paths.append("/home/lorenzo/model_editor_models")
    return gazebo_model_paths


def solve_uri(uri: str):
    if os.path.exists(uri):
        return uri
    model_name = uri.replace("model://", "")
    gazebo_model_paths = gazebo_models_folders()
    for gazebo_models_folder in gazebo_model_paths:
        path_candidate = os.path.join(gazebo_models_folder, model_name)
        if os.path.exists(path_candidate):
            return path_candidate


def parse_include_tag(include: Element):
    pose = include.find("pose")
    uri = include.find("uri").text
    path_to_file = solve_uri(uri)
    return ModelFileToTrimeshScene(path_to_file).scene


class ModelFileToTrimeshScene:
    def __init__(self, path_to_model_folder):
        self.path_to_world_file = path_to_model_folder
        for element in os.listdir(path_to_model_folder):
            if ".sdf" in element:
                path_to_model_file = os.path.join(path_to_model_folder, element)
        self.tf = {"": np.eye(4)}
        self.xml = parse(path_to_model_file)
        self.parent_map = {c: p for p in self.xml.iter() for c in p}
        self.scene = None


elements_to_print = dict()


def find_transformation_matrix_between_vectors(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    if np.linalg.norm(v1 - v2, 2) < 0.01:
        return np.eye(4)
    rotation_axis = np.cross(v1, v2)
    rotation_angle = np.arccos(np.dot(v1, v2))
    rotation_matrix = create_rotation_matrix(rotation_axis, rotation_angle)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    return transformation_matrix


def create_rotation_matrix(axis, angle):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    rotation_matrix = np.array([[t * x * x + c, t * x * y - s * z, t * x * z + s * y], [t * x * y + s * z, t * y * y + c, t * y * z - s * x], [t * x * z - s * y, t * y * z + s * x, t * z * z + c]])
    return rotation_matrix


def xyzrpytoT(*args):
    x, y, z, roll, pitch, yaw = args if len(args) == 6 else args[0]
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    rotation_matrix = np.array([[cy * cp, -sy * cr + cy * sp * sr, sy * sr + cy * sp * cr, 0], [sy * cp, cy * cr + sy * sp * sr, -cy * sr + sy * sp * cr, 0], [-sp, cp * sr, cp * cr, 0], [0, 0, 0, 1]])
    translation_matrix = np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
    transformation_matrix = np.dot(translation_matrix, rotation_matrix)
    return transformation_matrix


class WorldFileToTrimeshScene:
    def __init__(self, path_to_world_file) -> None:
        self.path_to_world_file = path_to_world_file
        self.xml = parse(path_to_world_file)
        self.parent_map = {c: p for p in self.xml.iter() for c in p}
        self.trimesh_scene = trimesh.Scene()
        self.parse()

    def parse(self):
        for element in self.xml.iter():
            if element.tag in ["box", "capsule", "cylinder", "elipsoid", "mesh", "plane"]:  # , "polyline", "sphere"]:
                self.parse_geometry(element)
            elif element.tag == "include":
                tree = parse_include_tag(element)

    def parse_geometry(self, geometry: Element):
        parent = self.parent_map[geometry]
        parents = [parent]
        is_collision = False
        while parent.tag != "world":
            parent = self.parent_map[parent]
            if parent.tag == "collision":
                is_collision = True or is_collision
            parents.append(parent)
        if not is_collision:
            return
        parent_poses = list()
        for parent in parents:
            if not parent.find("pose") is None:
                parent_poses.append(parent.find("pose"))
        current_tf = np.eye(4)
        for parent_pose in parent_poses:
            pose = [i for i in parent_pose.text.split(" ")]
            if len(pose) != 6:
                pose_temp = []
                for i in pose:
                    try:
                        pose_temp.append(float(i))
                    except:
                        pass
                pose = pose_temp
            pose = [float(i) for i in pose]
            current_tf = current_tf @ xyzrpytoT(pose)
        self.add_geometry_to_scene(geometry, current_tf)

    def add_geometry_to_scene(self, geometry: Element, tf):
        trimesh_g = None
        if geometry.tag == "plane":
            normal = np.array([float(i) for i in geometry.find("normal").text.split(" ")])
            T = find_transformation_matrix_between_vectors(np.array((0, 0, 1)), normal)
            trimesh_g = creation.box((20, 20, 0), transform=T)
        elif geometry.tag == "mesh":
            path_to_mesh = solve_uri(geometry.find("uri").text)
            scale = geometry.find("scale")
            if not path_to_mesh is None:
                trimesh_g = trimesh.load(path_to_mesh, force="mesh")
                if not scale is None:
                    scale = float(scale.text)
                    trimesh_g.apply_scale(scale)
        elif geometry.tag == "box":
            size = [float(i) for i in geometry.find("size").text.split(" ")]
            trimesh_g = creation.box(size)
        elif geometry.tag == "cylinder":
            radius = float(geometry.find("radius").text)
            height = float(geometry.find("length").text)
            trimesh_g = creation.cylinder(radius, height)
        elif geometry.tag == "polyline":
            pass
        if not trimesh_g is None:
            self.trimesh_scene.add_geometry(trimesh_g, transform=tf)


def main():
    base_folder = "/home/lorenzo/repos/gazebo_worlds/comprehensive_dataset_worlds/"
    for folder in os.listdir(base_folder):
        path_to_file = os.path.join(base_folder, folder, "world.world")
        scene = WorldFileToTrimeshScene(path_to_file)
        scene.trimesh_scene.add_geometry(creation.box())
        scene.trimesh_scene.camera_transform = np.eye(4)
        scene.trimesh_scene.show()


if __name__ == "__main__":
    main()
