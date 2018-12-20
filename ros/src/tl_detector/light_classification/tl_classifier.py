from styx_msgs.msg import TrafficLight

import tensorflow as tf
import numpy as np
import cv2

# confidence cut-off for detection
CONFIDENCE_CUTOFF = 0.8

# the number of classes (red, green, yellow)
NUM_CLASSES = 3


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

    def __init__(self, frozen_graph="frozen_inference_graph.pb"):

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

        with tf.Session(graph=self.detection_graph) as sess:

            # Actual detection.
            (boxes, scores, classes) = sess.run([self.detection_boxes, self.detection_scores,
                                                    self.detection_classes],
                                                    feed_dict={self.image_tensor: image_np})

            # Remove unnecessary dimensions
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)
            # Filter boxes with a confidence score less than `confidence_cutoff`
            boxes, scores, classes = self.filter_boxes(self.conf_cutoff, boxes, scores, classes)

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
            if(len(classes) > 0):
                max_id = list(scores).index(max(scores))
                tl_id = (int(classes[max_id]) + 1) % 3
            else: # if nothing detected, return unknown
                tl_id = TrafficLight.UNKNOWN

        return tl_id

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

# # for check
# import cv2
# if __name__ == '__main__':
#     tlc = TLClassifier()
#     image = cv2.imread('image2.jpg')
#     print(tlc.get_classification(image))
