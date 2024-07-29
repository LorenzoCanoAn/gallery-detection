import rospy
import matplotlib.pyplot as plt


def plot(image, prediction):
    plt.subplot(2, 1, 1)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 1, 2)
    plt.plot(prediction[0])
    plt.xticks([0, 90, 180, 270, 360], labels=["$0$", "$\\pi/2$", "$\\pi$", "$3\\pi/2$", "$2\\pi$"])
    plt.show()


class Node:
    def __init__(self):
        rospy.init_node("plot")
