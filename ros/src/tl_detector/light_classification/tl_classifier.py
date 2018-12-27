from styx_msgs.msg import TrafficLight

import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor
from PIL import ImageFont
import rospy
import datetime

import logging

# path to the frozen graph for the trained model
# PATH_TO_FROZEN_GRAPH = "frozen_inference_graph.pb"
PATH_TO_FROZEN_GRAPH = "light_classification/frozen_inference_graph.pb"
# confidence cut-off for detection
CONFIDENCE_CUTOFF = 0.8

# the number of classes (red, green, yellow)
NUM_CLASSES = 3

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data/', 'label_map_sdc.pbtxt')

# shoud match with the order in label_map_sdc.pbtxt
CLASSNAME_LIST = ['Green', 'Red', 'Yellow','Unknown'] # list of class name
COLOR_LIST = ['lawngreen', 'red', 'yellow'] # list of color to be used for visual purpose below


# NOTE
# For TrafficLight, mapping between color (UNKNOWN, GREEN, YELLOW and RED) and
# integer indices goes as follows (see src/styx_msgs/msg/TrafficLight.msg)
# TrafficLight.UNKNOWN = 4
# TrafficLight.GREEN = 2
# TrafficLight.YELLOW = 1
# TrafficLight.RED = 0
# On the other hand, the mapping in the tensorflow model goes as follows
# Green: 1, Red: 2, Yellow: 3


class TLClassifier(object):

    def __init__(self, frozen_graph="frozen_inference_graph.pb", debug=False):

        self.debug = debug

        logging.getLogger('tensorflow').disabled = True

        # path to the frozen graph for the trained model
        PATH_TO_FROZEN_GRAPH = "light_classification/" + frozen_graph

        # Load the neural network model for traffic light classification
        self.detection_graph = self.load_graph(PATH_TO_FROZEN_GRAPH)

        # The input placeholder for the image.
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        # The classification of the object (integer id).
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

        # cutoff for score of detection
        self.conf_cutoff = CONFIDENCE_CUTOFF
        
        self.tfsession = self.start_session()


    def start_session(self): 
        sess = tf.Session(graph=self.detection_graph)
        return sess


    def run_session(self,image_np):
        return self.tfsession.run([self.detection_boxes, self.detection_scores,
                                                    self.detection_classes],
                                                    feed_dict={self.image_tensor: image_np})


    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """

        # NOTE: here I assume that image is in np.array format
        cv_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # reshape for input to the trained neural network model
        image_np = np.expand_dims(cv_image, 0)

        # Actual detection + measuring execution time
        time_start = datetime.datetime.now()
        (boxes, scores, classes) = self.run_session(image_np)
        time_finish = datetime.datetime.now()

        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        # Filter boxes with a confidence score less than `confidence_cutoff`
        boxes, scores, classes = self.filter_boxes(self.conf_cutoff, boxes, scores, classes)

        cv_image_out_rgb = None
        rospy.logwarn("[tl_classifier] debug {0} ".format(self.debug))
        if self.debug:
            image_pil = Image.fromarray(cv_image)

            # The current box coordinates are normalized to a range between 0 and 1.
            # This converts the coordinates actual location on the image.
            width, height = image_pil.size
            box_coords = self.to_image_coords(boxes, height, width)

            # Each class with be represented by a differently colored box
            image_draw = self.draw_boxes(image_pil, box_coords, classes, scores)

            cv_image_out_bgr = np.array(image_draw.getdata(),np.uint8).reshape(image_draw.size[1], image_draw.size[0], 3)
            cv_image_out_rgb = cv2.cvtColor(cv_image_out_bgr, cv2.COLOR_BGR2RGB)             

        # # If doing majority vote, use the following code:
        # scores = []
        # classes =[]
        # if(len(classes) >0):
        #     classes = [int(cl) for cl in classes]
        #     class_count = []
        #     for id in range(1, 1+NUM_CLASSES):
        #         class_count.append(classes.count(id))
        #     max_count = max(class_count)
        #
        #     if(class_count.count(max_count) == 1): # if unique majority
        #         tl_id =  (class_count.index(max_count) + 1 + 1) % 3
        #     else: # if multiple classes tie, use the max score instead
        #         max_id = list(scores).index(max(scores))
        #         tl_id = (int(classes[max_id]) + 1) % 3
        # else:
        #     tl_id = TrafficLight.UNKNOWN

        # If simpliy choosing the one with the highest score use below.
        # If one or more lights detected, select the class with the highest score
        if len(classes) > 0:
            max_id = list(scores).index(max(scores))
            tl_id = (int(classes[max_id]) + 1) % 3

        else: # if nothing detected, return unknown
            tl_id = TrafficLight.UNKNOWN

        time_processing = time_finish - time_start
        rospy.logwarn("[tl_cllassifier] Inference : tl_id {0} Time: {1} ".format(CLASSNAME_LIST[tl_id-1], time_processing))
        return tl_id, cv_image_out_rgb


    def filter_boxes(self, min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)

        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]

        return filtered_boxes, filtered_scores, filtered_classes


    def load_graph(self, graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph


    def to_image_coords(self,boxes, height, width):
        """
        The original box coordinate output is normalized, i.e [0, 1].

        This converts it back to the original coordinate based on the image
        size.
        """
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width

        return box_coords


    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


    def draw_boxes(self,image, boxes, classes, scores, thickness=4):
        """Draw bounding boxes on the image"""
        image_draw = image.copy()
        draw = ImageDraw.Draw(image_draw)
        for i in range(len(boxes)):
            bot, left, top, right = boxes[i, ...]
            class_id = int(classes[i])
            color = COLOR_LIST[class_id-1]
            cls_name = CLASSNAME_LIST[class_id-1]
            percent = str(round(scores[i] * 100, 1))
            txt_display = cls_name + ": " + percent + "%"
            # print(class_id, cls_name, color, txt_display)
            # draw.rectangle([(left, top-15), (left+80, top-thickness)], fill= color)
            draw.rectangle([(left-2, bot-15), (left+80, bot)], fill= color)
            draw.line([(left, top), (left, bot), (right, bot), (right, top), (left, top)], width=thickness, fill=color)
            draw.text((left, bot-15), txt_display, fill="black")
        return image_draw


    def numpy_array_to_cvimage(self, new_image,num_rows, num_cols):
        #new_image = np.ndarray((3, num_rows, num_cols), dtype=int)
        new_image = new_image.astype(np.uint8)
        new_image_red, new_image_green, new_image_blue = new_image
        new_rgb = np.dstack([new_image_red, new_image_green, new_image_blue])
        return new_rgb

# # for check
# import cv2
# if __name__ == '__main__':
#     tlc = TLClassifier()
#     image = cv2.imread('image2.jpg')
#     print(tlc.get_classification(image))
