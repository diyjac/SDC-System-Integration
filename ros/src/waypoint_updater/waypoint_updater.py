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
STOPPING = 2
GO = 3
GOFROMSTOP = 4


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
        # self.restricted_speed = 10. # mph
        self.restricted_speed = 9.5 # mph
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
        self.state = INIT

        # start the loop
        self.loop()

    def loop(self):
        # remove 1.0 mph for safety margin...
        self.restricted_speed -= 1.0
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
        if self.current_linear_velocity == 0. and self.state != INIT:
            self.state = STOP

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

        if self.current_linear_velocity < 0.01:
            self.state = STOP

        braking_distance = 3*(self.restricted_speed+self.current_linear_velocity)
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
        elif tl_dist is None:
            self.state = GO
            # calculate normal trajectory
            for i in range(len(vptsx)):
                cte = polynomial([vptsx[i]])[0]
                self.final_waypoints[i].twist.twist.angular.z = cte
            velocity = self.final_waypoints[0].twist.twist.linear.x

        # we are stopped and more than 6 meters from a red light
        # elif tl_dist is not None and (self.state == STOP or self.state == GOFROMSTOP) and tl_dist > 2.:
        elif tl_dist is not None and tl_dist > 6.0:
            self.state = GOFROMSTOP
            # calculate start and stop trajectory
            if tl_dist > 10.0:
                wpx = [0, self.redtlwp//3, self.redtlwp//2, self.redtlwp, len(vptsx)]
                wpy = [self.restricted_speed*mps/2, self.restricted_speed*mps, self.restricted_speed*mps/2, 0.0, 0.0]
                poly2 = np.polyfit(np.array(wpx), np.array(wpy), 3)
                polynomial2 = np.poly1d(poly2)
            elif tl_dist > 8.0 and self.current_linear_velocity < 2.5:
                wpx = [0, self.redtlwp, len(vptsx)]
                wpy = [self.current_linear_velocity/10, -10.0, -10.0]
                poly2 = np.polyfit(np.array(wpx), np.array(wpy), 2)
                polynomial2 = np.poly1d(poly2)
            elif tl_dist > 7.0 and self.current_linear_velocity < 1.5:
                wpx = [0, self.redtlwp, len(vptsx)]
                wpy = [self.current_linear_velocity/8, -10.0, -10.0]
                poly2 = np.polyfit(np.array(wpx), np.array(wpy), 2)
                polynomial2 = np.poly1d(poly2)
            elif tl_dist > 6.0 and self.current_linear_velocity < 0.75:
                wpx = [0, self.redtlwp, len(vptsx)]
                wpy = [self.current_linear_velocity/5, -10.0, -10.0]
                poly2 = np.polyfit(np.array(wpx), np.array(wpy), 2)
                polynomial2 = np.poly1d(poly2)
            elif self.current_linear_velocity < 0.01:
                wpx = [0, self.redtlwp, len(vptsx)]
                wpy = [-0.01, -0.01, -0.01]
                poly2 = np.polyfit(np.array(wpx), np.array(wpy), 2)
                polynomial2 = np.poly1d(poly2)
            elif self.current_linear_velocity < 0.5:
                wpx = [0, self.redtlwp, len(vptsx)]
                wpy = [-0.5, -0.5, -0.5]
                poly2 = np.polyfit(np.array(wpx), np.array(wpy), 2)
                polynomial2 = np.poly1d(poly2)
            elif self.current_linear_velocity < 1.:
                wpx = [0, self.redtlwp, len(vptsx)]
                wpy = [-2., -2., -2.]
                poly2 = np.polyfit(np.array(wpx), np.array(wpy), 2)
                polynomial2 = np.poly1d(poly2)
            else:
                wpx = [0, self.redtlwp, len(vptsx)]
                wpy = [-10, -10, -10]
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

        elif tl_dist is not None and self.state != STOP:
            if self.current_linear_velocity < 0.5:
                wpx = [0, self.redtlwp, len(vptsx)]
                wpy = [-1., -1., -1.]
                poly2 = np.polyfit(np.array(wpx), np.array(wpy), 2)
                polynomial2 = np.poly1d(poly2)
            elif self.current_linear_velocity < 1.:
                wpx = [0, self.redtlwp, len(vptsx)]
                wpy = [-4., -4., -4.]
                poly2 = np.polyfit(np.array(wpx), np.array(wpy), 2)
                polynomial2 = np.poly1d(poly2)
            elif self.current_linear_velocity < 2.:
                wpx = [0, self.redtlwp, len(vptsx)]
                wpy = [-20., -20., -20.]
                poly2 = np.polyfit(np.array(wpx), np.array(wpy), 2)
                polynomial2 = np.poly1d(poly2)
            elif self.current_linear_velocity < 3.:
                wpx = [0, self.redtlwp, len(vptsx)]
                wpy = [-40., -40., -40.]
                poly2 = np.polyfit(np.array(wpx), np.array(wpy), 2)
                polynomial2 = np.poly1d(poly2)
            else:
                wpx = [0, self.redtlwp, len(vptsx)]
                wpy = [-80., -80., -80.]
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
        # elif tl_dist is not None and self.state != STOP:
        #     # red traffic light
        #     self.state = STOPPING
        #
        #    # calculate stopping trajectory
        #    if self.redtlwp > 15:
        #        wpx = [0, self.redtlwp//2, self.redtlwp, len(vptsx)]
        #        wpy = [self.current_linear_velocity/2, self.current_linear_velocity/3, 0.0, 0.0]
        #        poly2 = np.polyfit(np.array(wpx), np.array(wpy), 3)
        #        polynomial2 = np.poly1d(poly2)
        #    elif self.redtlwp > 2:
        #        wpx = [0, self.redtlwp, len(vptsx)]
        #        wpy = [self.current_linear_velocity/4, -10.0, -10.0]
        #        poly2 = np.polyfit(np.array(wpx), np.array(wpy), 2)
        #        polynomial2 = np.poly1d(poly2)
        #    else:
        #        wpx = [0, self.redtlwp, len(vptsx)]
        #        wpy = [-10, -10, -10]
        #        poly2 = np.polyfit(np.array(wpx), np.array(wpy), 2)
        #        polynomial2 = np.poly1d(poly2)
        #
        #    for i in range(len(vptsx)):
        #        cte = polynomial([vptsx[i]])[0]
        #        velocity = polynomial2([i])[0]
        #
        #        # need to create a new waypoint array so not to overwrite old one
        #        p = Waypoint()
        #        p.pose.pose.position.x = self.final_waypoints[i].pose.pose.position.x
        #        p.pose.pose.position.y = self.final_waypoints[i].pose.pose.position.y
        #        p.pose.pose.position.z = self.final_waypoints[i].pose.pose.position.z
        #        p.pose.pose.orientation.x = self.final_waypoints[i].pose.pose.orientation.x
        #        p.pose.pose.orientation.y = self.final_waypoints[i].pose.pose.orientation.y
        #        p.pose.pose.orientation.z = self.final_waypoints[i].pose.pose.orientation.z
        #        p.pose.pose.orientation.w = self.final_waypoints[i].pose.pose.orientation.w
        #        p.twist.twist.linear.x = velocity
        #        p.twist.twist.linear.y = 0.
        #        p.twist.twist.linear.z = 0.
        #        p.twist.twist.angular.x = 0.
        #        p.twist.twist.angular.y = 0.
        #        p.twist.twist.angular.z = cte
        #        self.final_waypoints[i] = p
        #
        #    velocity = polynomial2([0])[0]

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
                p.twist.twist.linear.x = -0.01
                p.twist.twist.linear.y = 0.
                p.twist.twist.linear.z = 0.
                p.twist.twist.angular.x = 0.
                p.twist.twist.angular.y = 0.
                p.twist.twist.angular.z = cte
                self.final_waypoints[i] = p

            velocity = 0

        print "state:", self.state, "current_linear_velocity:", self.current_linear_velocity, "velocity:", velocity, "redtlwp", self.redtlwp, "tl_dist:", tl_dist

    def waypoints_cb(self, msg):
        # DONE: Implement
        mps = 0.44704
        # make our own copy of the waypoints - they are static and do not change
        if self.waypoints is None:
            self.waypoints = []
            for waypoint in msg.waypoints:
                waypoint.twist.twist.linear.x = mps*self.restricted_speed
                self.waypoints.append(waypoint)

            # calculate number of waypoints for stopping using current restricted speed limit.
            wlen = len(self.waypoints)
            wstop = 20 # stop within twenty waypoints
            for i in range(wstop):
                self.waypoints[wlen-1-i].twist.twist.linear.x = (mps*self.restricted_speed*i)/wstop
            self.waypoints[wlen-1].twist.twist.linear.x = -0.01

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
        if self.state == INIT and self.i > 350:
            self.state = STOP

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
