
import cv2
import rospy
from time import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


cvbridge = CvBridge()
output_dir = './imgs/simulator_images/'
i = 0

def image_callback(msg):

    image_name = 'simulator_' + i + '.png' 
    
    #print('Writing image %s' % image_name)

    try:

        cv2_img = cvbridge.imgmsg_to_cv2(msg, "bgr8")


    except CvBridgeError, e:

        print(e)

    else:

        cv2.imwrite(output_dir + image_name, cv2_img)
    
    i++    



def main():

    rospy.init_node('image_extractor')

    image_topic = "/image_color"

    rospy.Subscriber(image_topic, Image, image_callback)

    rospy.spin()


if __name__ == '__main__':
    main()