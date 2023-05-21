#!/bin/python3
import pathlib
import torch
from cv_bridge import CvBridge
import sensor_msgs.msg as sensor_msg
import std_msgs.msg as std_msg
import rospy
import importlib
import numpy as np


class NetworkNode:
    def __init__(self):
        rospy.init_node(
            "gallery_network",
        )
        self.init_network()
        self._cv_bridge = CvBridge()
        self.image_subscriber = rospy.Subscriber(
            "/lidar_image_norm", sensor_msg.Image, self.image_callback
        )
        self.detection_publisher = rospy.Publisher(
            "/gallery_detection_vector", std_msg.Float32MultiArray, queue_size=10
        )

    def init_network(self):
        file_path = rospy.get_param(
            "~nn_path",
            default="/home/lorenzo/models/gallery_detection/procedural_datasets/dataset_03/gallery_detector_v3-_r10_lr002_4.torch",
        )
        print(file_path)
        file_name = pathlib.Path(file_path).name
        nn_type = file_name.split("-")[0]
        module = importlib.import_module("gallery_detection_models.models")
        self.model = getattr(module, nn_type)()
        rospy.set_param("/ptcl_to_img/image_width", self.model.input_width)
        rospy.set_param("/ptcl_to_img/image_height", self.model.input_height)
        self.model.load_state_dict(
            torch.load(file_path, map_location=torch.device("cpu"))
        )
        self.model.eval()

    def image_callback(self, msg: sensor_msg.Image):
        depth_image = np.reshape(
            np.frombuffer(msg.data, dtype=np.float32), (msg.height, msg.width)
        )
        depth_image_tensor = torch.tensor(depth_image).float().to(torch.device("cpu"))
        depth_image_tensor /= torch.max(depth_image_tensor)
        depth_image_tensor = torch.reshape(depth_image_tensor, [1, 1, 16, -1])
        data = self.model(depth_image_tensor)
        data = data.cpu().detach().numpy()
        data = data[0, :]
        dim = (std_msg.MultiArrayDimension("0", data.__len__(), 1),)
        layout = std_msg.MultiArrayLayout(dim, 0)
        output_message = std_msg.Float32MultiArray(layout, data.astype("float32"))
        self.detection_publisher.publish(output_message)


def main():
    network_node = NetworkNode()
    rospy.loginfo("Gallery network beguinning to spin")
    rospy.spin()


if __name__ == "__main__":
    main()
