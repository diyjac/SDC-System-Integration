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
GO = 1
MPS = 0.44704

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
        self.restricted_speed_in_mps = 0. # mps set it to zero at the beginnig until we get the /base_waypoints
        self.current_linear_velocity = 0.
        self.current_angular_velocity = 0.
        self.cruz_control = None
        self.cruzpoly = None
        self.decelpoly = None
        self.lights = []
        self.updateRate = 2 # update 2 times a second
        self.pose = None
        self.theta = None
        self.waypoints = None
        self.decels = None
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

    def distanceToTrafficLight(self):
        dist = None
        if self.redtlwp is not None:
            # red or yellow light - positive waypoints away
            if self.redtlwp > 0:
                dist = self.distance(self.waypoints, self.cwp, self.cwp+self.redtlwp)
            # green light - negative waypoints away
            elif self.redtlwp < -1:
                dist = -self.distance(self.waypoints, self.cwp, self.cwp-self.redtlwp)
        return dist

    # Transform from global Cartesian x,y to local car coordinates x,y
    # where lx is pointing to the positive x axis and ly is deviation from the car's path
    def getLocalXY(self, theta, x, y):
        # convert to local coordinates
        vx = x - self.position.x
        vy = y - self.position.y
        lx = vx*np.cos(self.theta) + vy*np.sin(self.theta)
        ly = -vx*np.sin(self.theta) + vy*np.cos(self.theta)
        return lx, ly

    # Transform from local Cartesian x,y to global car coordinates x,y
    # where lx is pointing to the positive x axis and ly is deviation from the car's path
    def getWorldXY(self, theta, lx, ly):
        # convert back to global coordinates
        x = lx*np.cos(theta) - ly*np.sin(theta) + self.position.x
        y = lx*np.sin(theta) + ly*np.cos(theta) + self.position.y
        return x, y

    def getWaypoints(self, number):
        self.final_waypoints = []
        vptsx = []
        vptsy = []
        vptsd = []
        wlen = len(self.waypoints)

        # still initializing - don't move yet.
        if self.state == INIT or self.cruzpoly is None or self.decelpoly is None:
            # calculate normal trajectory
            for i in range(len(vptsx)):
                # need to create a new waypoint array so not to overwrite old one - don't move
                p = Waypoint()
                p.pose.pose.position.x = self.waypoints[self.cwp+i].pose.pose.position.x
                p.pose.pose.position.y = self.waypoints[self.cwp+i].pose.pose.position.y
                p.pose.pose.position.z = self.waypoints[self.cwp+i].pose.pose.position.z
                p.pose.pose.orientation.x = self.waypoints[self.cwp+i].pose.pose.orientation.x
                p.pose.pose.orientation.y = self.waypoints[self.cwp+i].pose.pose.orientation.y
                p.pose.pose.orientation.z = self.waypoints[self.cwp+i].pose.pose.orientation.z
                p.pose.pose.orientation.w = self.waypoints[self.cwp+i].pose.pose.orientation.w
                p.twist.twist.linear.x = 0.
                p.twist.twist.linear.y = 0.
                p.twist.twist.linear.z = 0.
                p.twist.twist.angular.x = 0.
                p.twist.twist.angular.y = 0.
                p.twist.twist.angular.z = 0.
                self.final_waypoints.append(p)
            velocity = 0

        # GO!
        else:
            # change to local coordinates
            dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
            p0 = self.waypoints[self.cwp].pose.pose.position
            wpd = 0.
            for i in range(number):
                x = self.waypoints[(self.cwp+i)%wlen].pose.pose.position.x
                y = self.waypoints[(self.cwp+i)%wlen].pose.pose.position.y
                lx, ly = self.getLocalXY(self.theta, x, y)
                vptsx.append(lx)
                vptsy.append(ly)
                p1 = self.waypoints[(self.cwp+i)%wlen].pose.pose.position
                ld = dl(p0, p1)
                vptsd.append(wpd+ld)
                p0 = p1
                wpd += ld

            # calculate cross track error (cte) for steering
            steerpoly = np.polyfit(np.array(vptsx), np.array(vptsy), 3)
            polynomial = np.poly1d(steerpoly)

            # calculate cross track error (cte) for throttle/braking
            tl_dist = self.distanceToTrafficLight()

            # if green light and is less than 16 meters away - reset and GO!
            if tl_dist is not None and tl_dist < 0. and tl_dist > -16.:
                tl_dist = None

            # if less than LOOKAHEAD_WPS points from the end of the tracks
            if self.cwp > (wlen-LOOKAHEAD_WPS):
                # calculate distance from last waypoint - minus five meters.
                velpolypoint = self.distance(self.waypoints, self.cwp, wlen-2) - 5.

                # already stopped within 5 meters of end waypoint and velocity is less than 1 MPS...
                if velpolypoint < 5. and self.current_linear_velocity < MPS:
                    # calculate trajectory with stoppoly
                    self.cruz_control = self.stoppoly
                else:
                    # calculate trajectory with decelpoly (remember its backwards!)
                    self.cruz_control = self.decelpoly

            # if green traffic light but will take more than 2 seconds to reach it at current speed...
            elif tl_dist is not None and tl_dist < 0. and self.current_linear_velocity > MPS*5 and -tl_dist > self.current_linear_velocity*2:
                # calculate trajectory with decelpoly (remember its backwards!)
                self.cruz_control = self.decelpoly
                tl_dist = -tl_dist
                velpolypoint = tl_dist

            # if no traffic light or cannot stop in less than 16 meters and current velocity is over 6.5 Meters a second
            elif tl_dist is None or (tl_dist < 16. and self.current_linear_velocity > 6.5*MPS):
                # calculate trajectory with cruzpoly
                self.cruz_control = self.cruzpoly
                velpolypoint = 0.0

            # already stopped within 5 meters of a traffic light and velocity is less than 1 MPS...
            elif tl_dist is not None and tl_dist < 5. and self.current_linear_velocity < MPS:
                # calculate trajectory with stoppoly
                self.cruz_control = self.stoppoly
                velpolypoint = 0.0

            # traffic light coming near
            else:
                # calculate trajectory with decelpoly (remember its backwards!)
                self.cruz_control = self.decelpoly
                velpolypoint = tl_dist

            # calculate trajectory
            for i in range(len(vptsx)):
                scte = polynomial([vptsx[i]])[0]
                ideal_velocity = self.cruz_control([(velpolypoint-vptsd[i])])[0]

                # we are off by more than 2 meters per second!
                if self.current_linear_velocity > (ideal_velocity + 2.):
                    # make hard correction.
                    ideal_velocity *= 0.25

                # need to create a new waypoint array so not to overwrite old one
                p = Waypoint()
                p.pose.pose.position.x = self.waypoints[(self.cwp+i)%wlen].pose.pose.position.x
                p.pose.pose.position.y = self.waypoints[(self.cwp+i)%wlen].pose.pose.position.y
                p.pose.pose.position.z = self.waypoints[(self.cwp+i)%wlen].pose.pose.position.z
                p.pose.pose.orientation.x = self.waypoints[(self.cwp+i)%wlen].pose.pose.orientation.x
                p.pose.pose.orientation.y = self.waypoints[(self.cwp+i)%wlen].pose.pose.orientation.y
                p.pose.pose.orientation.z = self.waypoints[(self.cwp+i)%wlen].pose.pose.orientation.z
                p.pose.pose.orientation.w = self.waypoints[(self.cwp+i)%wlen].pose.pose.orientation.w
                p.twist.twist.linear.x = ideal_velocity
                p.twist.twist.linear.y = 0.
                p.twist.twist.linear.z = 0.
                p.twist.twist.angular.x = 0.
                p.twist.twist.angular.y = 0.
                p.twist.twist.angular.z = scte
                self.final_waypoints.append(p)

            # set ideal velocity for current waypoint
            velocity = self.cruz_control([(velpolypoint-vptsd[0])])[0]

        print "state:", self.state, "current_linear_velocity:", self.current_linear_velocity, "velocity:", velocity, "redtlwp", self.redtlwp, "tl_dist:", tl_dist, "restricted_speed:", self.restricted_speed_in_mps

    def waypoints_cb(self, msg):
        # DONE: Implement
        # make our own copy of the waypoints - they are static and do not change
        if self.waypoints is None:
            # unsubscribe to the waypoint messages - cut down on resource usage
            self.sub_waypoints.unregister()
            self.sub_waypoints = None

            # set the restricted speed limit
            self.restricted_speed_in_mps = msg.waypoints[0].twist.twist.linear.x

            # create our own copy of the waypoint array
            self.waypoints = []
            vptsd = []
            dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
            p0 = msg.waypoints[0].pose.pose.position
            wpd = 0.
            for waypoint in msg.waypoints:
                # calculate distances
                p1 = waypoint.pose.pose.position
                ld = dl(p0, p1)
                vptsd.append(wpd+ld)
                p0 = p1
                wpd += ld

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

            # use the restricted speed limit in mps to create our deceleration points (in reverse)
            wpx0 = [-LOOKAHEAD_WPS, 0., LOOKAHEAD_WPS]
            wpy0 = [self.restricted_speed_in_mps, self.restricted_speed_in_mps, self.restricted_speed_in_mps]
            poly0 = np.polyfit(np.array(wpx0), np.array(wpy0), 2)
            self.cruzpoly = np.poly1d(poly0)

            # since LOOKAHEAD_WPS is current set to 100 (meters 200*.5), a half is around 50 meters stopping distance
            wpx1 = []
            wpy1 = []

            wpx1.append(-LOOKAHEAD_WPS)
            wpy1.append(-0.1)

            wpx1.append(0.)
            wpy1.append(-0.2)

            # 5 meters away
            wpx1.append(5)
            wpy1.append(MPS*.5)

            wpx1.append(10)
            wpy1.append(MPS*5)

            wpx1.append(16)
            wpy1.append(MPS*5)

            # 2 seconds away
            wpx1.append(max([self.restricted_speed_in_mps*2, 24]))
            wpy1.append(max([self.restricted_speed_in_mps*.2, MPS*5]))

            # 4 seconds away
            wpx1.append(max([self.restricted_speed_in_mps*4, 45]))
            wpy1.append(max([self.restricted_speed_in_mps*.3, MPS*6]))

            # 6 seconds away
            wpx1.append(max([self.restricted_speed_in_mps*6, 65]))
            wpy1.append(max([self.restricted_speed_in_mps*.5, MPS*10]))

            # 8 seconds away
            wpx1.append(max([self.restricted_speed_in_mps*8, 85]))
	    wpy1.append(self.restricted_speed_in_mps)

            wpx1.append(LOOKAHEAD_WPS)
            wpy1.append(self.restricted_speed_in_mps)

            poly1 = np.polyfit(np.array(wpx1), np.array(wpy1), 3)
            self.decelpoly = np.poly1d(poly1)

            # use the -0.01 speed to create our stop points
            wpx2 = [-LOOKAHEAD_WPS, 0., LOOKAHEAD_WPS]
            wpy2 = [-0.01, -0.01, -0.01]
            poly2 = np.polyfit(np.array(wpx2), np.array(wpy2), 2)
            self.stoppoly = np.poly1d(poly2)

            wlen = len(self.waypoints)
            velpolypoint = vptsd[wlen-1]
            for i in range(LOOKAHEAD_WPS):
                ideal_velocity = self.decelpoly([(velpolypoint-vptsd[wlen-1-LOOKAHEAD_WPS+i])])[0]
                self.waypoints[wlen-1-LOOKAHEAD_WPS+i].twist.twist.linear.x = ideal_velocity
                

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
        self.redtlwp = msg.data
        if self.state == INIT and self.i > 200:
            self.state = GO

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
