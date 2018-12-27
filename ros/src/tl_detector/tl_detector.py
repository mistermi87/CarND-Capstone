#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import numpy as np

import datetime
from scipy.spatial import KDTree

STATE_COUNT_THRESHOLD = 3

# Skip certain number of images to relieve a developer machine (if needed).
# If set to False, every image will be used.
SKIP_IMAGES = 1

class TLDetector(object):

    def __init__(self):

        rospy.init_node('tl_detector')

        self.waypoints_2d = None
        self.waypoint_tree = None

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        # Conter for image skipping
        self.image_counter = 0
        self.listener = tf.TransformListener()
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1, buff_size=20*800*600*3)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        self.inference_image = rospy.Publisher('/inference_image',Image, queue_size=1)

        self.bridge = CvBridge()

        # Setting up the classifier
        frozen_graph = rospy.get_param('~frozen_graph', "frozen_inference_graph.pb")
        debug = rospy.get_param('~debug', "false")
        self.is_site = rospy.get_param('~is_site', "false")
        #self.is_site_param = rospy.get_param("/is_site")
        rospy.logwarn("[tl_detector] is_site {0} ".format(self.is_site))
        
        #rospy.logwarn("[tl_detector] Parameters {0} ".format(rospy.get_param_names()))
        
        self.light_classifier = TLClassifier(frozen_graph, debug)

        # Running a first inference so that the model gets fully loaded
        fake_image_data = np.zeros([600, 800, 3], np.uint8)
        _ = self.light_classifier.get_classification(fake_image_data)



        rospy.spin()


    def pose_cb(self, msg):
        self.pose = msg


    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
                self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints ]
                #rospy.logwarn("Waypoints_2d: {0}".format(self.waypoints_2d))
                time_start = datetime.datetime.now()
                self.waypoint_tree = KDTree(self.waypoints_2d)
                #rospy.logwarn("waypoint_tree: {0}".format(self.waypoint_tree))   
                time_finish = datetime.datetime.now()

                time_processing = time_finish - time_start
                rospy.logwarn("[tl_detector]: Time: {0} ".format(time_processing))     


    def traffic_cb(self, msg):
        self.lights = msg.lights


    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """

        # skipping images
        if SKIP_IMAGES != False:

            self.image_counter += 1
            if self.image_counter % (SKIP_IMAGES + 1) != 0:
                return

            # avoiding overflow in the long term: resetting the counter
            self.image_counter = 0

        self.has_image = True
        self.camera_image = msg

        light_wp, state = self.process_traffic_lights()
        #rospy.logwarn("Closest light wp: {0} light state {1}".format(light_wp, state))

        # We don't want to risk the counter reset on a yellow --> red state change (goodbye, TrafficLight.YELLOW)
        if state == TrafficLight.YELLOW:
            state = TrafficLight.RED

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state

        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED or state == TrafficLight.YELLOW else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))

        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))

        self.state_count += 1


    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement

        closest_idx = self.waypoint_tree.query([x,y],1)[1]
        return closest_idx


    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # # for testing
        # return light.state

        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        # Get classification
        tl_id, image_draw = self.light_classifier.get_classification(cv_image)
        
        self.inference_image.publish(self.bridge.cv2_to_imgmsg(image_draw, "bgr8"))
        
        return tl_id


    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        #TODO find the closest visible traffic light (if one exists)

        closest_light = None
        line_waypoint_idx = None

        stop_line_positions = self.config['stop_line_positions']
        if (self.pose):
            car_waypoint_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

            diff = len(self.waypoints.waypoints)
            for i , light in enumerate(self.lights):
                line = stop_line_positions[i]
                temp_waypoint_index = self.get_closest_waypoint(line[0], line[1])
                d = temp_waypoint_index - car_waypoint_idx
                if d >= 0 and d < diff:
                    diff = d
                    closest_light = light
                    line_waypoint_idx = temp_waypoint_index
        
        if closest_light:
            state = self.get_light_state(closest_light)
            return line_waypoint_idx, state

        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':

    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
