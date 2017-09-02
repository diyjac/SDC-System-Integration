from lowpass import LowPassFilter
from pid import PID

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


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
            self.pid = PID(2.0, 0.4, 0.1)

    def control(self, *args, **kwargs):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        if len(args) == 4:
            self.current_linear_velocity = args[2]
            if abs(self.current_linear_velocity) < 0.01:
                self.current_linear_velocity = 0.
            self.dbw_enabled = args[3]
            if self.dbw_enabled:
                self.desired_linear_velocity = args[0]
                self.desired_angular_velocity = args[1]
                steer = self.desired_angular_velocity * self.steer_ratio
                brake = 0.
                throttle = self.pid.step(self.desired_linear_velocity, self.sample_time)
                # throttle = self.lowpass.filt(self.pid.step(self.desired_linear_velocity, self.sample_time))
                if throttle < 0. and throttle < -self.brake_deadband:
                    brake = abs(throttle * self.vehicle_mass * self.wheel_radius)
                    throttle = 0.
                elif self.desired_linear_velocity < self.current_linear_velocity:
                    throttle = 0.
                    if self.current_linear_velocity <= self.min_speed:
                        brake = abs(self.vehicle_mass * self.wheel_radius)
                    if self.current_linear_velocity == 0 and self.desired_linear_velocity > 0.:
                        brake = 0.
                return throttle, brake, steer
            else:
                self.pid.reset()
        return 0., 0., 0.
