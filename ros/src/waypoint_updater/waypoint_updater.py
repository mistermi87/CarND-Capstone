#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from scipy.spatial import KDTree
import numpy as np
import math
import datetime
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

# Whether LOOKAHEAD_WPS and MAX_DECEL shoud be adjusted depending on the maximum allowed velocity
ADJUST_LOOKAHEAD = True
ADJUST_DECELERATION = True

# Number of waypoints we will publish. You can change this number
LOOKAHEAD_WPS = 10
MAX_DECEL = .5

class WaypointUpdater(object):

    def __init__(self):
        rospy.init_node('waypoint_updater')

        # Defining variables
        # self.base_waypoints = None      # for the simpler version of the waypoint updater
        self.pose = None
        self.base_lane = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.stopline_waypoint_idx = -1
        self.waypoint_distances = None
        self.waypoint_distances_precomputed = False

        # Subscribing to interesting topics
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)

        # Publishing interesting things
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        
        self.loop()


    def loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            #if self.pose and self.base_waypoints :
            if self.pose and self.waypoint_distances_precomputed: # self.base_lane:
                self.publish_waypoints()
                #closest_waypoint_idx = self.get_closest_waypoint_idx()
                #self.publish_waypoints(closest_waypoint_idx)
                
            rate.sleep()


    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x    
        y = self.pose.pose.position.y

        closest_idx = self.waypoint_tree.query([x,y],1)[1]   
        
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx-1]

        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x,y])

        val = np.dot(cl_vect - prev_vect , pos_vect - cl_vect)

        if val > 0:
            closest_idx = ( closest_idx + 1) % len(self.waypoints_2d)

        return closest_idx


    def publish_waypoints(self):
        #lane = Lane()
        #lane.header = self.base_waypoints.header
        #lane.waypoints = self.base_waypoints.waypoints[closest_idx:closest_idx+LOOKAHEAD_WPS]
        #self.final_waypoints_pub.publish(lane)
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)
        

    def generate_lane (self):
        lane = Lane()
        closest_idx = self.get_closest_waypoint_idx()
        furthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_lane.waypoints[closest_idx:furthest_idx]

        if self.stopline_waypoint_idx == -1 or (self.stopline_waypoint_idx >= furthest_idx):
            lane.waypoints = base_waypoints

        else:
            # rospy.logwarn("Decelerating. len(wps): {}, closest_idx: {}, stopline_waypoint_idx: {}".format(len(base_waypoints), closest_idx, self.stopline_waypoint_idx))
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)

        return lane


    def decelerate_waypoints(self, waypoints, closest_idx):
        temp = []
        for i, wp in enumerate(waypoints):

            p = Waypoint()
            p.pose = wp.pose

            stop_idx = max(self.stopline_waypoint_idx - closest_idx - 4, 0)

            # Calculating the distance only in case the target is still ahead of the current waypoint
            dist = 0
            if i < stop_idx:
                dist = self.distance(waypoints, i, stop_idx, closest_idx)

            vel = math.sqrt(2* MAX_DECEL * dist)
            if vel < 1.0:
                vel = 0
            
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)

        return temp


    def pose_cb(self, msg):
        # TODO: Implement
        self.pose = msg


    def waypoints_cb(self, waypoints):

        # TODO: Implement
        self.base_lane = waypoints
        if not self.waypoints_2d:

            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints ]
            time_start = datetime.datetime.now()
            self.waypoint_tree = KDTree(self.waypoints_2d)
            time_finish = datetime.datetime.now()

            time_processing = time_finish - time_start
            rospy.logwarn("[waypoint_updater]: Time: {0} ".format(time_processing))
            # Calculating the distance between each two consecutive waypoints and storing them
            sum_dist = 0
            self.waypoint_distances = []

            nr_of_segments = len(self.base_lane.waypoints) - 1
            for i in range(nr_of_segments):

                tmp_dist = self.distance(self.base_lane.waypoints, i, i+1)
                self.waypoint_distances.append(tmp_dist)
                sum_dist += tmp_dist

            # Calculating average distance + signalizing the end of the operation
            avg_wp_dist = sum_dist / nr_of_segments
            self.waypoint_distances_precomputed = True

            rospy.logwarn("[waypoints_cb] Total track length: {:.2f} meters over {} waypoints".format(sum_dist, nr_of_segments))
            rospy.logwarn("[waypoints_cb] Average distance between the waypoints: {:.3f} meters".format(avg_wp_dist))

            global LOOKAHEAD_WPS, MAX_DECEL
            max_vel_kmh = rospy.get_param('/waypoint_loader/velocity', 40.0)
            max_vel_ms = max_vel_kmh * 1000.0 / 3600.0
            rospy.logwarn("[waypoints_cb] Maximum allowed veloctiy: {:.2f} km/h ({:.2f} m/s)".format(max_vel_kmh, max_vel_ms))

            if ADJUST_DECELERATION:
                # Reference value: decelerating by 2 m/s^2 at 40 km/h
                # A deceleration value will be picked that is linearly proportional to the base value
                MAX_DECEL = max(1.0, min(3, max_vel_kmh / 20.0) )
                rospy.logwarn("[waypoints_cb] Using adjusted deceleration: {} m/s^2".format(MAX_DECEL))

            if ADJUST_LOOKAHEAD:
                # s = v^2 / 2*a
                # Calculating the distance that is needed for the vehicle to slow down from the maximum
                # allowed velocity to zero, using the given deceleration.
                # TODO: incorporate vehicle mass and weather and the color of the sky, etc.
                brake_distance = max_vel_ms ** 2 / (2*MAX_DECEL)
                # Then we calculate the number of waypoints which are along this distance _on average_
                # (...plus a few so that the vehicle can certainly stop within the calculated distance.)
                waypoint_count = int(math.ceil(brake_distance / avg_wp_dist))
                safety_reserve = int(math.ceil(max_vel_kmh / 10.0)) + 1
                LOOKAHEAD_WPS = max(15, min(150, waypoint_count + safety_reserve))
                rospy.logwarn("[waypoints_cb] Using adjusted lookahead: {} waypoints".format(LOOKAHEAD_WPS))


    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.stopline_waypoint_idx = msg.data


    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        self.stopline_waypoint_idx = msg.data


    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x


    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity


    def distance(self, waypoints, wp1, wp2, base_idx=0):
        dist = 0
        if self.waypoint_distances_precomputed:
            for i in range(wp1, wp2):
                dist += self.waypoint_distances[base_idx + i]

        else:
            dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
            for i in range(wp1, wp2+1):
                dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
                wp1 = i

        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
