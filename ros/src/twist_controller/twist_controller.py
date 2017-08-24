from lowpass import LowPassFilter
from pid import PID

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, *args, **kwargs):
        # DONE: Implement
        self.sample_time = 0.5
        if len(args) == 5:
            self.wheel_base = args[0]
            self.steer_ratio = args[1]
            self.min_speed = args[2]
            self.max_lat_accel = args[3]
            self.max_steer_angle = args[4]
            self.lowpass = LowPassFilter(self.max_lat_accel, self.sample_time)
            self.pid = PID(2.0, 0.4, 0.1)

    def control(self, *args, **kwargs):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        if len(args) == 4:
            self.dbw_enabled = args[3]
            if self.dbw_enabled:
                self.desired_linear_velocity = args[0]
                self.desired_angular_velocity = args[1]
                self.current_linear_velocity = args[2]
                steer = self.desired_angular_velocity * self.steer_ratio
                brake = 0.
                # throttle = self.lowpass.filt(self.pid.step(self.desired_linear_velocity, self.sample_time))
                throttle = self.pid.step(self.desired_linear_velocity, self.sample_time)
                if throttle < 0.:
                    throttle = 0.
                elif self.desired_linear_velocity < self.current_linear_velocity:
                    throttle = 0.
                    if self.current_linear_velocity <= self.min_speed:
                        brake = 1.
                    if self.current_linear_velocity == 0 and self.desired_linear_velocity > 0.:
                        brake = 0.
                return throttle, brake, steer
            else:
                self.pid.reset()
        return 0., 0., 0.
