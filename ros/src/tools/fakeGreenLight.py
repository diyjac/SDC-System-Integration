#!/usr/bin/env python

import argparse
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
import math
import sys
import numpy as np
import csv

class FakeGreenLight():
    def __init__(self):
        # initialize and subscribe to the current position and waypoint base topic
        rospy.init_node('fake_green_light')
        self.sub_current_pose = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg.pose
        self.upcoming_red_light_pub.publish(Int32(-1))


if __name__ == "__main__":
    try:
        FakeGreenLight()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not send fake green light message.')

