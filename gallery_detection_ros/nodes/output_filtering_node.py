# This node takes the output of the neural network as an imput, and outputs a list of angles at which a gallery could be present
import std_msgs.msg as std_msg
import rospy
import numpy as np
import math


def min_distance(angle, obj):
    distance = (angle - obj) % (math.pi * 2)
    if distance < -math.pi:
        distance += math.pi * 2
    elif distance > math.pi:
        distance -= math.pi * 2
    distance = abs(distance)
    return distance


def array_position_to_angle(array_position):
    return ((180 - array_position) / 180.0 * math.pi + 2 * math.pi) % (2 * math.pi)


def filtered_to_gallery_angles(filtered):
    max_peak = np.max(filtered)
    ratio = 0.3
    galleries_indices = np.nonzero(filtered > max_peak * ratio)[0]
    galleries_angles = []
    for index in galleries_indices:
        galleries_angles.append(array_position_to_angle(index))
    return galleries_angles, filtered[galleries_indices]


class FilteringNode:
    def __init__(self):
        rospy.init_node("gallery_vector_filtering")
        self.detection_publisher = rospy.Publisher(
            "/currently_detected_galleries", std_msg.Float32MultiArray, queue_size=10
        )

        rospy.Subscriber(
            "/gallery_detection_vector",
            std_msg.Float32MultiArray,
            callback=self.filter_vector,
        )

    def filter_vector(self, msg):
        vector = np.array(msg.data)
        vector = np.flip(vector)
        vector = np.roll(vector, 180)
        filtered = np.zeros(360)
        for i in range(360):
            to_check = vector[i]
            filtered[i] = to_check
            a = 40
            for j in range(a):
                index_inside_subsection = ((-int(a / 2) + j) + i) % 356
                if vector[index_inside_subsection] > to_check:
                    filtered[i] = 0
        gallery_angles, values = filtered_to_gallery_angles(filtered)
        dim = (std_msg.MultiArrayDimension("0", gallery_angles.__len__(), 2),)
        layout = std_msg.MultiArrayLayout(dim, 0)
        output_message = std_msg.Float32MultiArray(layout, np.hstack([gallery_angles, values]))

        self.detection_publisher.publish(output_message)


def main():
    filtering_node = FilteringNode()
    rospy.spin()


if __name__ == "__main__":
    main()
