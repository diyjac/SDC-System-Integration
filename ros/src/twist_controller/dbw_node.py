#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
import math

from twist_controller import Controller
from yaw_controller import YawController

#import pygame
#pygame.init()
#pygame.joystick.init()

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''

class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node')

        self.vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        self.fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        self.brake_deadband = rospy.get_param('~brake_deadband', .1)
        self.decel_limit = rospy.get_param('~decel_limit', -5)
        self.accel_limit = rospy.get_param('~accel_limit', 1.)
        self.wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        self.wheel_base = rospy.get_param('~wheel_base', 2.8498)
        self.steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        self.max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        self.max_steer_angle = rospy.get_param('~max_steer_angle', 8.)
        self.min_speed = rospy.get_param('~min_speed', 4.*0.44704)

        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd',
                                         SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd',
                                            ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd',
                                         BrakeCmd, queue_size=1)

        self.shutdown = False
        self.dbw_enabled = False
        self.current_velocity = 0.
        self.current_linear_velocity = 0.
        self.current_angular_velocity = 0.
        self.linear_velocity = 0.
        self.angular_velocity = 0.

        # TODO: Create `TwistController` object
        #self.controller = TwistController(
        #    self.wheel_base, self.steer_ratio, self.min_speed,
        #    self.max_lat_accel, self.max_steer_angle)

        self.controller = YawController(
            self.wheel_base, self.steer_ratio, self.min_speed,
            self.max_lat_accel, self.max_steer_angle)

        # TODO: Subscribe to all the topics you need to
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb)
        rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_cb)

        # go into event loop
        self.loop()

    def loop(self):
        rate = rospy.Rate(50) # 50Hz
        while not rospy.is_shutdown():
            # TODO: Get predicted throttle, brake, and steering using `twist_controller`
            # You should only publish the control commands if dbw is enabled
            # throttle, brake, steering = self.controller.control(<proposed linear velocity>,
            #                                                     <proposed angular velocity>,
            #                                                     <current linear velocity>,
            #                                                     <dbw status>,
            #                                                     <any other argument you need>)
            #for event in pygame.event.get():
            #    continue
            #self.joystick = pygame.joystick.Joystick(0)
            #self.joystick.init()
            #steer = -self.joystick.get_axis(0)
            #throttle = self.joystick.get_axis(1) + .3
            brake = 0.
            #if throttle < 0.:
            #    brake = -throttle
            #    throttle = 0.
            throttle = 0.5
            # steer = self.controller.get_steering(self.linear_velocity, self.angular_velocity, self.current_velocity)
            # steer = (self.controller.get_steering(self.linear_velocity, self.angular_velocity, self.current_velocity) + (self.angular_velocity - self.current_angular_velocity))
            steer = (self.angular_velocity - self.current_angular_velocity) * self.steer_ratio
            print "steer:", steer
            
            if self.dbw_enabled:
                self.publish(throttle, brake, steer)
                # self.publish(throttle, brake, -0.04)
            #else:
            #    # disabled!  Recreate/reset controller
            #    self.controller = TwistController(
            #        self.wheel_base, self.steer_ratio, self.min_speed,
            #        self.max_lat_accel, self.max_steer_angle)
            #    self.current_velocity = 0.
            #    self.linear_velocity = 0.
            #    self.angular_velocity = 0.

            rate.sleep()

    def dbw_enabled_cb(self, msg):
        self.dbw_enabled = msg

    def velocity_cb(self, msg):
        self.current_linear_velocity = msg.twist.linear.x
        self.current_angular_velocity = msg.twist.angular.z
        print "velocity: ", msg
        print ""

    def twist_cb(self, msg):
        self.twist_cmd = msg
        self.linear_velocity = self.twist_cmd.twist.linear.x
        self.angular_velocity = self.twist_cmd.twist.angular.z
        print "twist_cmd:", msg
        print ""

    def publish(self, throttle, brake, steer):
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)


if __name__ == '__main__':
    DBWNode()
