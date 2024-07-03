# This node takes the output of the neural network as an imput, and outputs a list of angles at which a gallery could be present
import std_msgs.msg as std_msg
import rospy
import numpy as np
import math
from gallery_detection_ros.msg import DetectionVector, DetectedGalleries, DetectionVectorStability


def min_distance(angle, obj):
    distance = (angle - obj) % (math.pi * 2)
    if distance < -math.pi:
        distance += math.pi * 2
    elif distance > math.pi:
        distance -= math.pi * 2
    distance = abs(distance)
    return distance


def array_position_to_angle(array_position):
    return np.deg2rad(array_position)


def filtered_to_gallery_angles(filtered, min_value):
    galleries_indices = np.nonzero(filtered > min_value)[0]
    galleries_angles = []
    for index in galleries_indices:
        galleries_angles.append(array_position_to_angle(index))
    values = filtered[galleries_indices]
    return (galleries_angles, values)


def is_max_in_window(array: np.ndarray, idx: int, width: int):
    to_check = array[idx]
    start_idx_w = -int(width / 2) + idx
    end_idx_w = start_idx_w + width
    window = np.take(array, np.r_[start_idx_w:end_idx_w], mode="wrap")
    if to_check >= np.max(window):
        return True
    else:
        return False


class FilteringNode:
    def __init__(self):
        rospy.init_node("gallery_vector_filtering")
        self.window_width = rospy.get_param(
            "~window_width", default=10
        )  # Width of window used to check if an element is the max
        self.threshold_to_detect = rospy.get_param(
            "~threshold_to_detect", default=0.4
        )  # If "max_value in a window is less that this fraction of the largest one in the vector, it is not considered"
        self.detection_publisher = rospy.Publisher(
            "/currently_detected_galleries", DetectedGalleries, queue_size=1
        )
        self.filtered_vector_publisher = rospy.Publisher(
            "/filtered_detection_vector", DetectionVector, queue_size=1
        )
        rospy.Subscriber(
            "/gallery_detection_vector",
            DetectionVector,
            callback=self.filter_vector,
        )

    def filter_vector(self, msg: DetectionVector):
        vector = np.array(msg.vector)
        header = msg.header
        filtered = np.zeros(360)
        for i in range(len(vector)):
            filtered[i] = vector[i] if is_max_in_window(vector, i, self.window_width) else 0
        filtered_vector_msg = DetectionVector(header, filtered)
        gallery_angles, gallery_values = filtered_to_gallery_angles(
            filtered, self.threshold_to_detect
        )
        detected_galleries_msg = DetectedGalleries(header, gallery_angles, gallery_values)
        self.filtered_vector_publisher.publish(filtered_vector_msg)
        self.detection_publisher.publish(detected_galleries_msg)


def main():
    filtering_node = FilteringNode()
    rospy.spin()


if __name__ == "__main__":
    main()
