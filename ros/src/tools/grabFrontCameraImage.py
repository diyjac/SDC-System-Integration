#!/usr/bin/env python

import argparse
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs
import cv2
import sys
import numpy as np
import image_geometry
from sensor_msgs.msg import CameraInfo
from traffic_light_config import config

class GrabFrontCameraImage():
    def __init__(self, outfile):
        # initialize and subscribe to the camera image and traffic lights topic
        rospy.init_node('front_camera_image_grabber')
        self.outfile = outfile

        self.cv_image = None
        self.lights = []

        sub2 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub3 = rospy.Subscriber('/camera/image_raw', Image, self.image_cb)

        self.pub = rospy.Publisher('/tl_debug', Image, queue_size=10)


        self.bridge = CvBridge()
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def traffic_cb(self, msg):
        self.lights = msg.lights


    def transform_3d_2d(self, point):
        x = 400 - 2560* config.camera_info.focal_length_x * point[0] / point[2]
        y =   600 - 1280 * config.camera_info.focal_length_y * point[1] / point[2] 
        return (x,y)

    def project_to_image_plane(self, pose):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            pose (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """
        point_in_world = pose.pose.position

        fx = config.camera_info.focal_length_x
        fy = config.camera_info.focal_length_y

        image_width = config.camera_info.image_width
        image_height = config.camera_info.image_height




        # get transform between pose of camera and world frame
        trans = None
        try:
            transform =  self.tfBuffer.lookup_transform("base_link","world",rospy.Time(0),rospy.Duration(1.0))

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logerr("Failed to find camera to map transform")

        #TODO Use tranform and rotation to calculate 2D position of light in image

      
        wp = np.array([ point_in_world.x, point_in_world.y, point_in_world.z ])

        pose_transformed = tf2_geometry_msgs.do_transform_pose(pose,transform)
        point = pose_transformed.pose.position

        if(point.x > 0 and point.x < 200 and abs(point.y) < 200 ): #light is infront
            print "point_in_world: ", wp
            print "pose_transformed: ", pose_transformed.pose.position

            point1 = np.array([point.y+1,point.z+1 ,point.x ])
            point2 = np.array([point.y -1,point.z - 2 ,point.x ])

            (x,y) = self.transform_3d_2d(point1)
            (x1,y1) = self.transform_3d_2d(point2)

            print "x, y:", x, y
            return (x,y,x1,y1)
        else:
            return (-1,-1,-1,-1)

    def draw_light_box(self, light):
        """Draw boxes around traffic lights

        Args:
            light (TrafficLight): light to classify

        Returns:
            image with boxes around traffic lights

        """
        (x,y,x1,y1) = self.project_to_image_plane(light.pose)

        # use light location to draw box around traffic light in image

        x = int(x)
        y = int(y)
        x1 = int(x1)
        y1 = int(y1)
        if(x > 0 and y > 0 and x < config.camera_info.image_width and y < config.camera_info.image_height):

            color = (0,0,0)
            if(light.state == 0):
                color = (0,0,255)
            elif(light.state == 1):
                color = (0,255,255)
            elif(light.state == 2):
                color = (0,255,0)
            print "state:", light.state
            cv2.rectangle(self.cv_image,(x, y),(x1,y1),color,3)

    def image_cb(self, msg):
        """Grab the first incoming camera image and saves it

        Args:
            msg (Image): image from car-mounted camera

        """
        if len(self.lights) > 0:
            height = int(msg.height)
            width = int(msg.width)
            msg.encoding = "rgb8"
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # TODO: experiment with drawing bounding boxes around traffic lights
            for light in self.lights:
                self.draw_light_box(light)
                
            im_pub = self.bridge.cv2_to_imgmsg(self.cv_image)
            self.pub.publish(im_pub)
            #cv2.imwrite(self.outfile, self.cv_image)


            #rospy.signal_shutdown("grabbed image done.")
            #sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Udacity SDC System Integration, Front Camera Image Grabber')
    parser.add_argument('outfile', type=str, help='Image save file')
    args = parser.parse_args()

    try:
        GrabFrontCameraImage(args.outfile)
    except rospy.ROSInterruptException:
        rospy.logerr('Could not grab front camera image.')


