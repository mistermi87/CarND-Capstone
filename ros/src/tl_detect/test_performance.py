
#--------------- import ----------------
import numpy as np
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor
from PIL import ImageFont
# import time
import tensorflow as tf
from scipy.stats import norm

print("tensorflow version:", tf.VERSION)

# -------------- Model preparation -----------

# What model to load
MODEL_NAME = 'training/model1/trained_model'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data/', 'label_map_sdc.pbtxt')

# shoud match with the order in label_map_sdc.pbtxt
CLASSNAME_LIST = ['Green', 'Red', 'Yellow'] # list of class name
COLOR_LIST = ['lawngreen', 'red', 'yellow'] # list of color to be used for visual purpose below

# path to test image directory
PATH_TO_TEST_IMAGES_DIR = 'data/test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 4) ]

# --------------- Load Frozen Tensorflow Model into Memory. -------------------

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ----------------- Helper Code -------------------------

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def filter_boxes(min_score, boxes, scores, classes):
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

def to_image_coords(boxes, height, width):
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

def draw_boxes(image, boxes, classes, scores, thickness=4):
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

def load_graph(graph_file):
    """Loads a frozen inference graph"""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph

# --------------- Detection ----------------------
detection_graph = load_graph(PATH_TO_FROZEN_GRAPH)

# The input placeholder for the image.
# `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Each box represents a part of the image where a particular object was detected.
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represent how level of confidence for each of the objects.
# Score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')

# The classification of the object (integer id).
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')


for image_path in TEST_IMAGE_PATHS:

    # Load a sample image.
    image = Image.open(image_path)
    image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)

    with tf.Session(graph=detection_graph) as sess:
        # Actual detection.
        (boxes, scores, classes) = sess.run([detection_boxes, detection_scores, detection_classes],
                                            feed_dict={image_tensor: image_np})

        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        confidence_cutoff = 0.8 # 0.8
        # Filter boxes with a confidence score less than `confidence_cutoff`
        boxes, scores, classes = filter_boxes(confidence_cutoff, boxes, scores, classes)

        # The current box coordinates are normalized to a range between 0 and 1.
        # This converts the coordinates actual location on the image.
        width, height = image.size
        box_coords = to_image_coords(boxes, height, width)

        # Each class with be represented by a differently colored box
        image_draw = draw_boxes(image, box_coords, classes, scores)

        # image_draw.show()
        save_image_path= image_path[:-4]+"detect.jpg"
        image_draw.save(save_image_path)
