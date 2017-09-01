#!/usr/bin/env python

import rospy
import tf
from geometry_msgs.msg import TwistStamped, PoseStamped
from styx_msgs.msg import TrafficLightArray, TrafficLight, Lane, Waypoint
from std_msgs.msg import Int32
import numpy as np
import copy

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number

# states
INIT = 0
STOP = 1
STOPPING1 = 2
STOPPING2 = 3
GO = 4
GOFROMSTOP = 5


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)
        self.sub_waypoints = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        # NOTE: comment out until we get traffic lights working...
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        # rospy.Subscriber('/obstacle_waypoint', Waypoint, self.obstacle_cb)

        # TODO: For testing - comment out when we have /traffic_waypoint working...
        # rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_light_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # DONE: Add other member variables you need below
        self.restricted_speed = 0. # mph set it to zero at the beginnig until we get the /base_waypoints
        self.current_linear_velocity = 0.
        self.current_angular_velocity = 0.
        self.lights = []
        self.updateRate = 2 # update 2 times a second
        self.pose = None
        self.theta = None
        self.waypoints = None
        self.cwp = None
        self.redtlwp = None
        self.i = 0
        self.go_timer = 0
        self.state = INIT

        # start the loop
        self.loop()

    def loop(self):
        # remove 1.0 mph for safety margin...
        rate = rospy.Rate(self.updateRate)
        while not rospy.is_shutdown():
            if self.waypoints and self.theta and self.state != INIT:
                self.cwp = self.nextWaypoint()
                self.getWaypoints(LOOKAHEAD_WPS)
                self.publish()
            rate.sleep()

    def pose_cb(self, msg):
        # DONE: Implement
        self.i += 1
        self.pose = msg.pose
        self.position = self.pose.position
        self.orientation = self.pose.orientation
        euler = tf.transformations.euler_from_quaternion([
            self.orientation.x,
            self.orientation.y,
            self.orientation.z,
            self.orientation.w])
        self.theta = euler[2]
        if self.state == INIT:
            print "INITIALIZING TRAFFIC LIGHT DETECTOR...."

    def velocity_cb(self, msg):
        self.current_linear_velocity = msg.twist.linear.x
        self.current_angular_velocity = msg.twist.angular.z
        if self.current_linear_velocity < 0.01 and self.state != INIT and self.state != GOFROMSTOP and self.state != STOP:
            self.state = GO

    def nextWaypoint(self, location=None):
        if location is None:
            location = self.position
        dist = 100000.
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        cwp = 0
        for i in range(len(self.waypoints)):
            d1 = dl(location, self.waypoints[i].pose.pose.position)
            if dist > d1:
                cwp = i
                dist = d1
        x = self.waypoints[cwp].pose.pose.position.x
        y = self.waypoints[cwp].pose.pose.position.y
        heading = np.arctan2((y-location.y), (x-location.x))
        angle = np.abs(self.theta-heading)
        if angle > np.pi/4.:
            cwp += 1
            if cwp >= len(self.waypoints):
                cwp = 0
        return cwp

    def distanceToREDTrafficLight(self):
        dist = None
        if self.redtlwp is not None:
            if self.redtlwp > 0:
                dist = self.distance(self.waypoints, self.cwp, self.cwp+self.redtlwp)
        return dist

    def getWaypoints(self, number):
        self.final_waypoints = []
        vptsx = []
        vptsy = []
        wlen = len(self.waypoints)

        # change to local coordinates
        for i in range(number):
            vx = self.waypoints[(i+self.cwp)%wlen].pose.pose.position.x - self.position.x
            vy = self.waypoints[(i+self.cwp)%wlen].pose.pose.position.y - self.position.y
            lx = vx*np.cos(self.theta) + vy*np.sin(self.theta)
            ly = -vx*np.sin(self.theta) + vy*np.cos(self.theta)
            vptsx.append(lx)
            vptsy.append(ly)
            self.final_waypoints.append(self.waypoints[(i+self.cwp)%wlen])

        # calculate cross track error (cte) + current lateral acceleration
        poly = np.polyfit(np.array(vptsx), np.array(vptsy), 3)
        polynomial = np.poly1d(poly)
        mps = 0.44704
        velocity = 0.

        if self.go_timer > 0:
            self.go_timer -= 1

        if self.current_linear_velocity < 0.01 and self.state != GOFROMSTOP:
            self.state = STOP

        braking_distance = self.restricted_speed*(self.current_linear_velocity/self.restricted_speed) + 8.
        tl_dist = self.distanceToREDTrafficLight()

        # still initializing - don't move yet.
        if self.state == INIT:
            # calculate normal trajectory
            for i in range(len(vptsx)):
                cte = polynomial([vptsx[i]])[0]

                # need to create a new waypoint array so not to overwrite old one
                p = Waypoint()
                p.pose.pose.position.x = self.final_waypoints[i].pose.pose.position.x
                p.pose.pose.position.y = self.final_waypoints[i].pose.pose.position.y
                p.pose.pose.position.z = self.final_waypoints[i].pose.pose.position.z
                p.pose.pose.orientation.x = self.final_waypoints[i].pose.pose.orientation.x
                p.pose.pose.orientation.y = self.final_waypoints[i].pose.pose.orientation.y
                p.pose.pose.orientation.z = self.final_waypoints[i].pose.pose.orientation.z
                p.pose.pose.orientation.w = self.final_waypoints[i].pose.pose.orientation.w
                p.twist.twist.linear.x = 0.
                p.twist.twist.linear.y = 0.
                p.twist.twist.linear.z = 0.
                p.twist.twist.angular.x = 0.
                p.twist.twist.angular.y = 0.
                p.twist.twist.angular.z = cte
                self.final_waypoints[i] = p

            velocity = 0

        # if no traffic light 
        elif tl_dist is None and self.state != INIT and self.state != GOFROMSTOP:
            if self.state != GO:
                print "previous state:", self.state
                self.go_timer = 20
            self.state = GO
            # calculate normal trajectory
            for i in range(len(vptsx)):
                cte = polynomial([vptsx[i]])[0]
                self.final_waypoints[i].twist.twist.angular.z = cte

        # we are stopped and more than a variable braking distance from a red light
        elif tl_dist is not None and tl_dist > 6 and tl_dist < 20 and self.state == GOFROMSTOP:
            # calculate start and stop trajectory
            wpx = [0, self.redtlwp//5, self.redtlwp//3, self.redtlwp, len(vptsx)]
            wpy = [5*mps, 5*mps, .0, -0.01, -0.01]
            poly2 = np.polyfit(np.array(wpx), np.array(wpy), 2)
            polynomial2 = np.poly1d(poly2)

            # calculate normal trajectory
            for i in range(len(vptsx)):
                cte = polynomial([vptsx[i]])[0]
                velocity = polynomial2([i])[0]

                # need to create a new waypoint array so not to overwrite old one
                p = Waypoint()
                p.pose.pose.position.x = self.final_waypoints[i].pose.pose.position.x
                p.pose.pose.position.y = self.final_waypoints[i].pose.pose.position.y
                p.pose.pose.position.z = self.final_waypoints[i].pose.pose.position.z
                p.pose.pose.orientation.x = self.final_waypoints[i].pose.pose.orientation.x
                p.pose.pose.orientation.y = self.final_waypoints[i].pose.pose.orientation.y
                p.pose.pose.orientation.z = self.final_waypoints[i].pose.pose.orientation.z
                p.pose.pose.orientation.w = self.final_waypoints[i].pose.pose.orientation.w
                p.twist.twist.linear.x = velocity
                p.twist.twist.linear.y = 0.
                p.twist.twist.linear.z = 0.
                p.twist.twist.angular.x = 0.
                p.twist.twist.angular.y = 0.
                p.twist.twist.angular.z = cte
                self.final_waypoints[i] = p

            velocity = polynomial2([0])[0]


        elif tl_dist is not None and tl_dist > braking_distance and self.go_timer == 0:
            self.state = STOPPING1
            # calculate start and stop trajectory

            braking = -2*(self.current_linear_velocity/self.restricted_speed*mps)*self.restricted_speed*(tl_dist - braking_distance)/tl_dist
            # braking += -self.current_linear_velocity*self.restricted_speed*(braking_distance/tl_dist)

            wpx = [0, self.redtlwp, len(vptsx)]
            wpy = [braking, 2*braking, 3*braking]
            poly2 = np.polyfit(np.array(wpx), np.array(wpy), 2)
            polynomial2 = np.poly1d(poly2)

            for i in range(len(vptsx)):
                cte = polynomial([vptsx[i]])[0]
                velocity = polynomial2([i])[0]

                # need to create a new waypoint array so not to overwrite old one
                p = Waypoint()
                p.pose.pose.position.x = self.final_waypoints[i].pose.pose.position.x
                p.pose.pose.position.y = self.final_waypoints[i].pose.pose.position.y
                p.pose.pose.position.z = self.final_waypoints[i].pose.pose.position.z
                p.pose.pose.orientation.x = self.final_waypoints[i].pose.pose.orientation.x
                p.pose.pose.orientation.y = self.final_waypoints[i].pose.pose.orientation.y
                p.pose.pose.orientation.z = self.final_waypoints[i].pose.pose.orientation.z
                p.pose.pose.orientation.w = self.final_waypoints[i].pose.pose.orientation.w
                p.twist.twist.linear.x = velocity
                p.twist.twist.linear.y = 0.
                p.twist.twist.linear.z = 0.
                p.twist.twist.angular.x = 0.
                p.twist.twist.angular.y = 0.
                p.twist.twist.angular.z = cte
                self.final_waypoints[i] = p

            velocity = polynomial2([0])[0]

        elif tl_dist is not None and self.state != STOP and self.go_timer == 0:
            self.state = STOPPING2
            braking = -2*self.current_linear_velocity*self.restricted_speed
            wpx = [0, self.redtlwp, len(vptsx)]
            wpy = [braking, 2*braking, 3*braking]
            poly2 = np.polyfit(np.array(wpx), np.array(wpy), 2)
            polynomial2 = np.poly1d(poly2)

            for i in range(len(vptsx)):
                cte = polynomial([vptsx[i]])[0]
                velocity = polynomial2([i])[0]

                # need to create a new waypoint array so not to overwrite old one
                p = Waypoint()
                p.pose.pose.position.x = self.final_waypoints[i].pose.pose.position.x
                p.pose.pose.position.y = self.final_waypoints[i].pose.pose.position.y
                p.pose.pose.position.z = self.final_waypoints[i].pose.pose.position.z
                p.pose.pose.orientation.x = self.final_waypoints[i].pose.pose.orientation.x
                p.pose.pose.orientation.y = self.final_waypoints[i].pose.pose.orientation.y
                p.pose.pose.orientation.z = self.final_waypoints[i].pose.pose.orientation.z
                p.pose.pose.orientation.w = self.final_waypoints[i].pose.pose.orientation.w
                p.twist.twist.linear.x = velocity
                p.twist.twist.linear.y = 0.
                p.twist.twist.linear.z = 0.
                p.twist.twist.angular.x = 0.
                p.twist.twist.angular.y = 0.
                p.twist.twist.angular.z = cte
                self.final_waypoints[i] = p

        # test if we are stopping for a traffic light
        elif tl_dist is not None and self.state == STOP:
            # calculate normal trajectory
            for i in range(len(vptsx)):
                cte = polynomial([vptsx[i]])[0]

                # need to create a new waypoint array so not to overwrite old one
                p = Waypoint()
                p.pose.pose.position.x = self.final_waypoints[i].pose.pose.position.x
                p.pose.pose.position.y = self.final_waypoints[i].pose.pose.position.y
                p.pose.pose.position.z = self.final_waypoints[i].pose.pose.position.z
                p.pose.pose.orientation.x = self.final_waypoints[i].pose.pose.orientation.x
                p.pose.pose.orientation.y = self.final_waypoints[i].pose.pose.orientation.y
                p.pose.pose.orientation.z = self.final_waypoints[i].pose.pose.orientation.z
                p.pose.pose.orientation.w = self.final_waypoints[i].pose.pose.orientation.w
                p.twist.twist.linear.x = 0.
                p.twist.twist.linear.y = 0.
                p.twist.twist.linear.z = 0.
                p.twist.twist.angular.x = 0.
                p.twist.twist.angular.y = 0.
                p.twist.twist.angular.z = cte
                self.final_waypoints[i] = p

            velocity = 0

        print "state:", self.state, "current_linear_velocity:", self.current_linear_velocity, "velocity:", velocity, "redtlwp", self.redtlwp, "tl_dist:", tl_dist, "braking_distance: ", braking_distance, "go_timer:", self.go_timer

    def waypoints_cb(self, msg):
        # DONE: Implement
        mps = 0.44704

        # set the restricted speed limit
        self.restricted_speed = msg.waypoints[0].twist.twist.linear.x/mps

        # make our own copy of the waypoints - they are static and do not change
        if self.waypoints is None:
            self.waypoints = []
            for waypoint in msg.waypoints:
                # need to create a new waypoint array so not to overwrite old one
                p = Waypoint()
                p.pose.pose.position.x = waypoint.pose.pose.position.x
                p.pose.pose.position.y = waypoint.pose.pose.position.y
                p.pose.pose.position.z = waypoint.pose.pose.position.z
                p.pose.pose.orientation.x = waypoint.pose.pose.orientation.x
                p.pose.pose.orientation.y = waypoint.pose.pose.orientation.y
                p.pose.pose.orientation.z = waypoint.pose.pose.orientation.z
                p.pose.pose.orientation.w = waypoint.pose.pose.orientation.w
                p.twist.twist.linear.x = waypoint.twist.twist.linear.x
                p.twist.twist.linear.y = waypoint.twist.twist.linear.y
                p.twist.twist.linear.z = waypoint.twist.twist.linear.z
                p.twist.twist.angular.x = waypoint.twist.twist.angular.x
                p.twist.twist.angular.y = waypoint.twist.twist.angular.y
                p.twist.twist.angular.z = waypoint.twist.twist.angular.z
                self.waypoints.append(p)

            # calculate number of waypoints for stopping using current restricted speed limit.
            wlen = len(self.waypoints)
            # stop within eighty waypoints
            wstop = 80
            for i in range(wstop):
                self.waypoints[wlen-20-i].twist.twist.linear.x = -1.0
            for i in range(20):
                self.waypoints[wlen-1-i].twist.twist.linear.x = -0.01

            # NOTE: Per Yousuf Fauzan, we should STOP at last waypoint, so commenting out
            #       the following code to do the wrap around for a looped course.
            #
            # make sure we wrap and correct for angles!
            # lastx = self.waypoints[len(msg.waypoints)-1].pose.pose.position.x
            # lasty = self.waypoints[len(msg.waypoints)-1].pose.pose.position.y
            # self.waypoints.append(msg.waypoints[0])
            # x = self.waypoints[len(msg.waypoints)-1].pose.pose.position.x
            # y = self.waypoints[len(msg.waypoints)-1].pose.pose.position.y
            # self.waypoints[len(self.waypoints)-1].twist.twist.angular.z = np.arctan2((y-lasty), (x-lastx))
            # lastx = x
            # lasty = y
            # self.waypoints.append(msg.waypoints[1])
            # x = self.waypoints[len(msg.waypoints)-1].pose.pose.position.x
            # y = self.waypoints[len(msg.waypoints)-1].pose.pose.position.y
            # self.waypoints[len(self.waypoints)-1].twist.twist.angular.z = np.arctan2((y-lasty), (x-lastx))

            # unsubscribe to the waypoint messages - cut down on resource usage
            self.sub_waypoints.unregister()
            self.sub_waypoints = None


    def traffic_cb(self, msg):
        # DONE: Callback for /traffic_waypoint message. Implement
        self.redtlwp = msg.data
        if self.state == INIT and self.i > 200:
            self.state = GOFROMSTOP

    def traffic_light_cb(self, msg):
        self.lights = msg.lights

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def publish(self):
        lane = Lane()
        lane.header.frame_id = '/world'
        lane.header.stamp = rospy.Time(0)
        lane.waypoints = self.final_waypoints
        self.final_waypoints_pub.publish(lane)


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
