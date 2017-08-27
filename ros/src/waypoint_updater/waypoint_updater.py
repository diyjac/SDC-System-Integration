#!/usr/bin/env python

import rospy
import tf
from geometry_msgs.msg import TwistStamped, PoseStamped
from styx_msgs.msg import TrafficLightArray, TrafficLight, Lane, Waypoint
import numpy as np
import copy

import math
from traffic_light_config import config

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
STOP = 0
STOPPING = 1
SLOWING = 2
GO = 3
GOGO = 4


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        # NOTE: comment out until we get traffic lights working...
        # rospy.Subscriber('/traffic_waypoint', Waypoint, self.traffic_cb)
        # rospy.Subscriber('/obstacle_waypoint', Waypoint, self.obstacle_cb)

        # TODO: For testing - comment out when we have /traffic_waypoint working...
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)

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
        self.i = 0
        self.state = STOP

        # start the loop
        self.loop()

    def loop(self):
        # remove 1.0 mph for safety margin...
        self.restricted_speed -= 1.0
        rate = rospy.Rate(self.updateRate)
        while not rospy.is_shutdown():
            if self.waypoints and self.theta:
                self.cwp = self.nextWaypoint()
                print self.i, "self.cwp:", self.cwp
                print ""
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

    def velocity_cb(self, msg):
        self.current_linear_velocity = msg.twist.linear.x
        self.current_angular_velocity = msg.twist.angular.z
        if self.current_linear_velocity == 0.:
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

    def distanceToNextTrafficLight(self):
        mps = 0.44704
        dist = 100000.
        dl = lambda a, b: math.sqrt((a.x-b[0])**2 + (a.y-b[1])**2)
        ctl = 0
        for i in range(len(config.light_positions)):
            d1 = dl(self.position, config.light_positions[i])
            if dist > d1:
                ctl = i
                dist = d1
        x = config.light_positions[ctl][0]
        y = config.light_positions[ctl][1]
        heading = np.arctan2((y-self.position.y), (x-self.position.x))
        angle = np.abs(self.theta-heading)
        if angle > np.pi/4.:
            ctl += 1
            if ctl >= len(config.light_positions):
                ctl = 0
            dist = dl(self.position, config.light_positions[ctl])
        self.ctl = ctl
        return dl(self.position, config.light_positions[self.ctl])

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


        # test if we are stopping for a traffic light
        braking_distance = 3*(self.restricted_speed+self.current_linear_velocity)
        tl_dist = self.distanceToNextTrafficLight()
        if (self.state == STOP or self.state == STOPPING) and tl_dist < braking_distance:
            if len(self.lights) > 0 and self.lights[self.ctl].state != 0 and self.current_linear_velocity == 0.:
                self.state = GOGO
        if self.state != STOP and self.state != GOGO and tl_dist < braking_distance:
            # red traffic light
            if len(self.lights) > 0 and self.lights[self.ctl].state == 0 and self.current_linear_velocity > 0.01:
                # stopping velocity
                if self.state == STOPPING and tl_dist < 2.:
                    if self.current_linear_velocity > mps:
                        velocity = -self.restricted_speed*mps*4
                    elif self.current_linear_velocity > mps/4:
                        velocity = -self.restricted_speed*mps
                    else:
                        velocity = 0.
                        self.state = STOP
                elif self.state == STOPPING and tl_dist < 5.:
                    if self.current_linear_velocity > 5*mps:
                        velocity = -self.restricted_speed*mps*4
                    elif self.current_linear_velocity > 3*mps:
                        velocity = -self.restricted_speed*mps*3
                    elif self.current_linear_velocity > 2*mps:
                        velocity = -self.restricted_speed*mps*2
                    elif self.current_linear_velocity > mps:
                        velocity = -self.restricted_speed*mps
                    else:
                        velocity = 1.*mps/5
                #elif self.state == STOPPING and tl_dist < 10. and self.current_linear_velocity > 3.:
                #    velocity = -self.restricted_speed
                #elif self.state == STOPPING and tl_dist < 10. and self.current_linear_velocity > 2.:
                #    velocity = -self.restricted_speed*.75
                elif self.state == STOPPING and tl_dist < 10. and self.current_linear_velocity > 1*mps:
                    velocity = 1.*mps/5
                #elif self.state == STOPPING and tl_dist < braking_distance*.25:
                #    if self.current_linear_velocity > 0.75:
                #        velocity = -self.restricted_speed*.1
                #    else:
                #        velocity = 0.
                #        self.state = STOP
                else:
                    if self.current_linear_velocity > 5*mps:
                        velocity = (self.restricted_speed*mps)/(tl_dist-(2*braking_distance/3))
                        self.state = STOPPING
                    else:
                        velocity = self.restricted_speed*mps/2.
                        self.state = STOPPING
            elif len(self.lights) > 0 and self.lights[self.ctl].state == 0 and self.current_linear_velocity < 0.01:
                velocity = 0.
                self.state = STOP
            elif len(self.lights) > 0 and self.lights[self.ctl].state != 0 and self.current_linear_velocity < 0.01:
                velocity = self.restricted_speed*mps
                self.state = GO
            else:
                # slow down near traffic lights
                velocity = self.restricted_speed*mps*.5
                self.state = SLOWING
            # calculate stopping trajectory
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
                p.twist.twist.linear.x = velocity
                p.twist.twist.linear.y = 0.
                p.twist.twist.linear.z = 0.
                p.twist.twist.angular.x = 0.
                p.twist.twist.angular.y = 0.
                p.twist.twist.angular.z = cte
                self.final_waypoints[i] = p
        else:
            self.state = GO
            # calculate normal trajectory
            for i in range(len(vptsx)):
                cte = polynomial([vptsx[i]])[0]
                if self.final_waypoints[i].twist.twist.linear.x < 0.:
                    self.final_waypoints[i].twist.twist.linear.x += 100.
                self.final_waypoints[i].twist.twist.angular.z = cte
        print "state:", self.state, "current_linear_velocity:", self.current_linear_velocity, "velocity:", velocity

    def waypoints_cb(self, msg):
        # DONE: Implement
        mps = 0.44704
        if self.waypoints is None:
            self.waypoints = []
            for waypoint in msg.waypoints:
                waypoint.twist.twist.linear.x = mps*self.restricted_speed
                self.waypoints.append(waypoint)

            # calculate number of waypoints for stopping using current restricted speed limit.
            wstop = int(4*self.restricted_speed)
            wlen = len(self.waypoints)
            # gradually zero out the last wstop waypoints (negative flag)
            for i in range(wstop):
                self.waypoints[wlen-1-i].twist.twist.linear.x = -100. + mps*self.restricted_speed*(i/wstop)
            # zero out the last waypoint
            self.waypoints[wlen-1].twist.twist.linear.x = -100.

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

    def traffic_cb(self, msg):
        # DONE: Callback for /traffic_waypoint message. Implement
        self.lights = msg.lights
        self.lights[1].state = 0

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
