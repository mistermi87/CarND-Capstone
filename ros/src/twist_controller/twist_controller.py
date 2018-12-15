
GAS_DENSITY = 2.858
ONE_MPH = 0.44704
import rospy
from pid import PID
from yaw_controller import YawController
from lowpass import LowPassFilter


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity,
                 brake_deadband,decel_limit,accel_limit,
                 wheel_radius,wheel_base,steer_ratio,
                 max_lat_accel,max_steer_angle):
        # TODO: Implement
        
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)

        kp = 0.3
        ki = 0.1
        kd = 0.
        mn = 0.
        mx = max(0.1, min(1.0, 0.2 * rospy.get_param('/waypoint_loader/velocity', 40.0) / 40.0))
        self.throttle_controller = PID(kp,ki,kd,mn,mx)

        tau = 0.5
        ts = 0.02

        self.vel_lpf = LowPassFilter(tau, ts)

        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        self.last_time = rospy.get_time()
        

    def control(self, current_velocity, dbw_enabled, linear_velocity, angular_velocity):
    
        # Return throttle, brake, steer
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0.,0.,0.
        #rospy.logwarn("current_velocity: {0} linear_velocity: {1} angular_velocity: {2}".format(current_velocity, linear_velocity , angular_velocity)) 
        current_velocity = self.vel_lpf.filt(current_velocity)
        
        #rospy.logwarn("Angular vel: {0}".format(angular_velocity))
        #rospy.logwarn("Target velocity: {0}".format(linear_velocity))
        #rospy.logwarn("Target angular velocity: {0}\n".format(angular_velocity))
        #rospy.logwarn("Current velocity: {0}\n".format(current_velocity))
        #rospy.logwarn("Filtered velocity: {0}\n".format(self.vel_lpf.get()))

        steering = self.yaw_controller.get_steering(linear_velocity, angular_velocity, current_velocity)

        vel_error = linear_velocity - current_velocity
        self.last_vel = current_velocity

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        throttle = self.throttle_controller.step(vel_error, sample_time)
        brake = 0

        if linear_velocity == 0. and current_velocity < 0.1:
            throttle = 0
            brake = 700 #N*m

        elif throttle < .1 and vel_error < 0:
            throttle = 0
            decel = max(vel_error, self.decel_limit)
            brake = abs(decel)*self.vehicle_mass*self.wheel_radius # TORQUE N*m
        #rospy.logwarn("throttle: {0} brake: {1} steering: {2}".format(throttle, brake , steering)) 
        return throttle, brake , steering