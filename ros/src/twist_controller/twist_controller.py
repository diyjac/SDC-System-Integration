from lowpass import LowPassFilter
from pid import PID
import numpy as np

GAS_DENSITY = 2.858
ONE_MPH = 0.44704
max_throttle_percentage = 0.8
max_brake_percentage = 0.8


class Controller(object):
    def __init__(self, *args, **kwargs):
        # DONE: Implement
        self.sample_time = 0.02
        if len(args) == 8:
            self.wheel_base = args[0]
            self.steer_ratio = args[1]
            self.min_speed = args[2]
            self.accel_limit = args[3]
            self.max_steer_angle = args[4]
            self.vehicle_mass = args[5]
            self.wheel_radius = args[6]
            self.brake_deadband = args[7]
            self.lowpass = LowPassFilter(self.accel_limit, self.sample_time)
            self.pid = PID(2.0, 0.4, 0.1, mn=-0.8, mx=0.8)

    def control(self, *args, **kwargs):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        if len(args) == 4:
            self.dbw_enabled = args[3]
            if self.dbw_enabled:
                self.ideal_linear_velocity = args[0]
                self.desired_angular_velocity = args[1]
                self.current_linear_velocity = args[2]
                steer = self.desired_angular_velocity * self.steer_ratio
                brake = 0.
                throttle = 0.
                # convert current velocity to ideal velocity delta percentage
                # throttle = np.max([-1.0, np.min([1.0, self.accel_limit, 2.*((self.ideal_linear_velocity-self.current_linear_velocity)/self.ideal_linear_velocity)])])
                if abs(self.ideal_linear_velocity) > abs(self.current_linear_velocity):
                    if self.ideal_linear_velocity < 0.:
                        throttle = -0.01
                    else:
                        factor = self.ideal_linear_velocity
                        throttle = np.max([np.min([4*(self.ideal_linear_velocity-self.current_linear_velocity+0.1)/factor, max_throttle_percentage]), -max_brake_percentage])
                    # throttle = self.pid.step(((self.ideal_linear_velocity-self.current_linear_velocity)/factor), self.sample_time)
                    # throttle = 2.*(self.ideal_linear_velocity-self.current_linear_velocity)/factor
                elif self.current_linear_velocity > 0.1:
                    factor = self.current_linear_velocity
                    throttle = np.max([np.min([4*(self.ideal_linear_velocity-self.current_linear_velocity-0.1)/factor, max_throttle_percentage]), -max_brake_percentage])
                else:
                    throttle = -0.01
                if throttle < 0.:
                    brake = -throttle
                    throttle = 0.
                return throttle, brake, steer
            else:
                self.pid.reset()
        return 0., 0., 0.
