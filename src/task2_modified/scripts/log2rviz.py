import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped
import os, sys


class Log2Rviz():
    def __init__(self):
        rospy.init_node("log2rviz_node")
        self._car1_pub = rospy.Publisher("visual/car1", Marker, queue_size=2)
        self._car2_pub = rospy.Publisher("visual/car2", Marker, queue_size=2)
        self._car3_pub = rospy.Publisher("visual/car3", Marker, queue_size=2)
        self._car4_pub = rospy.Publisher("visual/car4", Marker, queue_size=2)
        self._obj_pub = rospy.Publisher("visual/obj", Marker, queue_size=2)
        self._obs_pub = rospy.Publisher("visual/obs", Marker, queue_size=2)

        self._car1_sub = rospy.Subscriber("/vrpn_client_node/CAR1Final/pose", PoseStamped, self.car1_callback, queue_size=2)
        self._car2_sub = rospy.Subscriber("/vrpn_client_node/CAR2Final/pose", PoseStamped, self.car2_callback, queue_size=2)
        self._car3_sub = rospy.Subscriber("/vrpn_client_node/CAR3Final/pose", PoseStamped, self.car3_callback, queue_size=2)
        self._car4_sub = rospy.Subscriber("/vrpn_client_node/CAR4Final/pose", PoseStamped, self.car4_callback, queue_size=2)
        self._obj_sub = rospy.Subscriber("/vrpn_client_node/OBJ/pose", PoseStamped, self.obj_callback, queue_size=2)
        self._obs_sub = rospy.Subscriber("/vrpn_client_node/obstacle/pose", PoseStamped, self.obs_callback, queue_size=2)

        rospy.spin()
    
    def car_pub(self, msg: PoseStamped, publisher: rospy.Publisher):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.pose.position.x = msg.pose.position.x / 1000
        marker.pose.position.y = msg.pose.position.y / 1000
        marker.pose.position.z = msg.pose.position.z / 1000
        marker.pose.orientation = msg.pose.orientation
        marker.type = Marker.CUBE
        marker.scale.x = 0.26
        marker.scale.y = 0.18
        marker.scale.z = 0.06
        marker.color.g = 1
        marker.color.a = 1
        publisher.publish(marker)

    def car1_callback(self, msg: PoseStamped):
        self.car_pub(msg, self._car1_pub)
        
    def car2_callback(self, msg: PoseStamped):
        self.car_pub(msg, self._car2_pub)

    def car3_callback(self, msg: PoseStamped):
        self.car_pub(msg, self._car3_pub)

    def car4_callback(self, msg: PoseStamped):
        self.car_pub(msg, self._car4_pub)

    def obj_callback(self, msg: PoseStamped):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.pose.position.x = msg.pose.position.x / 1000
        marker.pose.position.y = msg.pose.position.y / 1000
        marker.pose.position.z = msg.pose.position.z / 1000 + 0.2
        marker.pose.orientation = msg.pose.orientation
        marker.type = Marker.CYLINDER
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.02
        marker.color.r = 1
        marker.color.a = 1
        self._obj_pub.publish(marker)

    def obs_callback(self, msg: PoseStamped):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.pose.position.x = msg.pose.position.x / 1000
        marker.pose.position.y = msg.pose.position.y / 1000
        marker.pose.position.z = msg.pose.position.z / 1000 + 0.1
        marker.pose.orientation = msg.pose.orientation
        marker.type = Marker.CYLINDER
        marker.scale.x = 0.35
        marker.scale.y = 0.35
        marker.scale.z = 0.2
        marker.color.b = 1
        marker.color.a = 1
        self._obs_pub.publish(marker)

if __name__ == "__main__":
    log2rviz = Log2Rviz()