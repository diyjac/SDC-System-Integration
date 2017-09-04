#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import math
import numpy as np
import tf
import cv2
from traffic_light_config import config

label = ['RED', 'YELLOW', 'GREEN', '', 'UNKNOWN']

STATE_COUNT_THRESHOLD = 3
LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. (reduce our waypoint search).

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.theta = None
        self.waypoints = None
        self.row = 600
        self.col = 800
        self.wlen = 0
        self.camera_image = None
        self.lights = []
        self.updateRate = 3 # 10Hz
        self.nwp = None
        self.traffic_light_to_waypoint_map = []

        self.sub_current_pose = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        self.sub_waypoints = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights helps you acquire an accurate ground truth data source for the traffic light
        classifier, providing the location and current color state of all traffic lights in the
        simulator. This state can be used to generate classified images or subbed into your solution to
        help you work on another single component of the node. This topic won't be available when
        testing your solution in real life so don't rely on it in the final submission.
        '''
        self.sub_raw_image = None

        self.upcoming_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier(rospy.get_param('~model_path'))

        self.init = True
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.ntlwp = None
        self.sub_raw_image = rospy.Subscriber('/camera/image_raw', Image, self.image_cb)

        # don't spin - control our resource usage!
        self.loop()

    def loop(self):
        # throttle our traffic light lookup until we are within range
        rate = rospy.Rate(self.updateRate)
        while not rospy.is_shutdown():
            if not self.init:
                if self.waypoints and self.theta:
                    self.nwp = self.nextWaypoint(self.pose)
                    self.ntlwp = self.getNextLightWaypoint(LOOKAHEAD_WPS)
                    if self.ntlwp is not None and self.sub_raw_image is None:
                        self.sub_raw_image = rospy.Subscriber('/camera/image_raw', Image, self.image_cb)
                    elif self.ntlwp is None and self.sub_raw_image is not None:
                        self.sub_raw_image.unregister()
                        self.sub_raw_image = None
                        self.last_wp = -1
                        self.upcoming_light_pub.publish(Int32(self.last_wp))
            rate.sleep()

    def pose_cb(self, msg):
        self.pose = msg.pose
        self.position = self.pose.position
        self.orientation = self.pose.orientation
        euler = tf.transformations.euler_from_quaternion([
            self.orientation.x,
            self.orientation.y,
            self.orientation.z,
            self.orientation.w])
        self.theta = euler[2]
        if self.light_classifier.predict is None:
            print "NOT MOVING!   Initializing TRAFFIC LIGHT DETECTOR...."

    def waypoints_cb(self, msg):
        # make our own copy of the waypoints - they are static and do not change
        if self.waypoints is None:
            self.waypoints = []
            for waypoint in msg.waypoints:
                self.waypoints.append(waypoint)
        self.wlen = len(self.waypoints)

        # only get it once - reduce resource consumption
        self.sub_waypoints.unregister()
        self.sub_waypoints = None

        # initialize lights to waypoint map
        self.initializeLightToWaypointMap()

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        self.row = int(msg.height)
        self.col = int(msg.width)
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            if state == TrafficLight.GREEN:
                light_wp = -light_wp
            elif state == TrafficLight.UNKNOWN:
                light_wp = -1
            self.last_wp = light_wp
            self.upcoming_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1
        if self.init:
            self.init = False

    def initializeLightToWaypointMap(self):
        # find the closest waypoint to the given (x,y) of the triffic light
        dl = lambda a, b: math.sqrt((a.x-b[0])**2 + (a.y-b[1])**2)
        for lidx in range(len(config.light_positions)):
            dist = 100000.
            tlwp = 0
            for widx in range(len(self.waypoints)):
                d1 = dl(self.waypoints[widx].pose.pose.position, config.light_positions[lidx])
                if dist > d1:
                    tlwp = widx
                    dist = d1
            self.traffic_light_to_waypoint_map.append(tlwp)

    def nextWaypoint(self, pose):
        """Identifies the next path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the next waypoint in self.waypoints

        """
        #DONE implement
        location = pose.position
        dist = 100000.
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        nwp = 0
        for i in range(len(self.waypoints)):
            d1 = dl(location, self.waypoints[i].pose.pose.position)
            if dist > d1:
                nwp = i
                dist = d1
        x = self.waypoints[nwp].pose.pose.position.x
        y = self.waypoints[nwp].pose.pose.position.y
        heading = np.arctan2((y-location.y), (x-location.x))
        angle = np.abs(self.theta-heading)
        if angle > np.pi/4.:
            nwp += 1
            if nwp >= len(self.waypoints):
                nwp = 0
        return nwp

    def getNextLightWaypoint(self, number):
        # find the closest waypoint from our pre-populated waypoint to traffic light map
        tlwp = None
        light = None
        for i in range(len(self.traffic_light_to_waypoint_map)):
            # make sure its forward in our direction
            if self.nwp < self.traffic_light_to_waypoint_map[i] and tlwp is None:
                tlwp = self.traffic_light_to_waypoint_map[i] - self.nwp

                # is it within the given number?
                if tlwp < number-2:
                    # is it within our traffic light tracking distance of 100 meters?
                    if self.distance(self.waypoints, self.nwp, (self.nwp+tlwp)%self.wlen) < 100.:
                        # set the traffic light waypoint target
                        # light = (self.nwp+tlwp)%self.wlen
                        # use relative waypoint ahead of current one instead!
                        light = tlwp
        return light

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return TrafficLight.RED

        self.camera_image.encoding = "rgb8"
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")
        if self.row != 600 or self.col != 800:
            image = cv2.resize(cv_image, (800, 600), interpolation=cv2.INTER_AREA)
        else:
            image = np.copy(cv_image)

        #Get classification
        classification = self.light_classifier.get_classification(image)
        print "traffic light: ", label[classification]
        return classification

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #DONE find the closest visible traffic light (if one exists within LOOKAHEAD_WPS)
        if self.init:
            state = self.get_light_state(0)
            return -1, TrafficLight.UNKNOWN
        elif self.ntlwp:
            state = self.get_light_state(self.ntlwp)
            # state = TrafficLight.RED
            return self.ntlwp, state
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
