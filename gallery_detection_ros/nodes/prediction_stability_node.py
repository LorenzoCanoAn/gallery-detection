import rospy
import std_msgs.msg as std_msg
from gallery_detection_ros.msg import DetectionVector, DetectionVectorStability
import numpy as np


def is_local_minima(array: np.ndarray, idx: int):
    return array[idx] < array[(idx - 1) % len(array)] and array[idx] < array[(idx + 1) % len(array)] and array[idx] > 0


def is_local_maxima(array: np.ndarray, idx: int):
    return array[idx] > array[(idx - 1) % len(array)] and array[idx] > array[(idx + 1) % len(array)] and array[idx] > 0


class Queue:
    def __init__(self, length):
        self.data = list()
        self.length = length

    def add_data(self, data):
        if len(self.data) < self.length:
            self.data.append(data)
        else:
            self.data.pop(0)
            self.data.append(data)

    def mean(self):
        return np.mean(np.array(self.data))

    def max(self):
        return np.max(np.array(self.data))

    def asarray(self):
        return np.array(self.data)


def has_local_minima(vector: np.ndarray):
    for idx in range(len(vector)):
        if is_local_minima(vector, idx) and vector[idx] > 0.1:
            return True
    return False


def local_maxima_avg(vector: np.ndarray):
    assert len(vector.shape) == 1
    sum_of_maxima = 0
    n_maxima = 0
    for idx in range(len(vector)):
        if is_local_maxima(vector, idx):
            sum_of_maxima += vector[idx]
            n_maxima += 1
    return sum_of_maxima / n_maxima


class LocalMaximaMagnitudeTracker:
    def __init__(self):
        self.prev_local_max_avg = None
        self.current_local_max_avg = None

    def new_vector(self, vector):
        if not self.current_local_max_avg is None:
            self.prev_local_max_avg = self.current_local_max_avg
        self.current_local_max_avg = local_maxima_avg(vector)

    def difference(self):
        if self.prev_local_max_avg is None:
            return 0
        else:
            return abs(self.prev_local_max_avg - self.current_local_max_avg)


class PredictionStabilityNode:
    def __init__(self):
        rospy.init_node(self.__class__.__name__)
        # Set variables
        self.stability_queue = Queue(8)
        self.avg_of_maxima_tracker = LocalMaximaMagnitudeTracker()
        # Get params
        self.prediction_vector_topic = rospy.get_param("~prediction_vector_topic")
        self.output_topic = rospy.get_param("~output_topic", "/is_detection_stable")
        # Set publishers
        self.output_publisher = rospy.Publisher(self.output_topic, DetectionVectorStability, queue_size=1)
        # Set subscribers
        rospy.Subscriber(
            self.prediction_vector_topic,
            DetectionVector,
            callback=self.prediction_vector_callback,
        )

    def prediction_vector_callback(self, msg: DetectionVector):
        vector = np.array(msg.vector).reshape(-1)
        has_loc_min = has_local_minima(vector)
        self.avg_of_maxima_tracker.new_vector(vector)
        is_stable = not has_loc_min  # and self.avg_of_maxima_tracker.difference() < 0.05
        self.stability_queue.add_data(is_stable)
        self.output_publisher.publish(DetectionVectorStability(msg.header, np.all(self.stability_queue.asarray())))

    def run(self):
        rospy.spin()


def main():
    node = PredictionStabilityNode()
    node.run()


if __name__ == "__main__":
    main()
