import pathlib
import torch
import cv2
from cv_bridge import CvBridge
import sensor_msgs.msg as sensor_msg
import std_msgs.msg as std_msg
from gallery_detection_ros.msg import DetectionVector
import rospy
import importlib
import numpy as np

from gallery_detection_models.models import GalleryDetectorV3

# model = GalleryDetectorV3(debug=True)
# model.load_state_dict(
#    torch.load(
#        "/media/lorenzo/SAM500/models/gallery-detection/GalleryDetectorV3.v2.torch",
#        map_location=torch.device("cuda"),
#    ),
# )
# model.eval()
# model.to("cuda")
# model(torch.rand((1, 1, 16, 720)).to("cuda"))
# exit()


class NetworkNode:
    def __init__(self):
        rospy.init_node(
            "gallery_network",
        )
        self.init_network()
        self._cv_bridge = CvBridge()
        self.image_subscriber = rospy.Subscriber("input_image_topic", sensor_msg.Image, self.image_callback)
        self.detection_publisher = rospy.Publisher("output_prediction_topic", DetectionVector, queue_size=1)

    def init_network(self):
        file_path = rospy.get_param("~nn_path")
        print(f"Loading{file_path}")
        file_name = pathlib.Path(file_path).name
        nn_type = file_name.split(".")[0]
        module = importlib.import_module("gallery_detection_models.models")
        self.model = getattr(module, nn_type)()
        self.model.load_state_dict(torch.load(file_path,weights_only=True))
        self.model.eval()

    def image_callback(self, msg: sensor_msg.Image):
        depth_image_raw = np.reshape(np.frombuffer(msg.data, dtype=np.float32), (msg.height, msg.width))
        depth_image_norm = depth_image_raw / np.max(depth_image_raw)
        depth_image_tensor = torch.tensor(depth_image_norm).float().to(torch.device("cpu"))
        depth_image_tensor = torch.reshape(depth_image_tensor, [1, 1, 16, -1])
        data = self.model(depth_image_tensor)
        data = np.array(data.cpu().detach().numpy())
        data = data.reshape(-1)
        output_message = DetectionVector(msg.header, data)
        self.detection_publisher.publish(output_message)


def main():
    network_node = NetworkNode()
    rospy.loginfo("Gallery network beguinning to spin")
    rospy.spin()


if __name__ == "__main__":
    main()
