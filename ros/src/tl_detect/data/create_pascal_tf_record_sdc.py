"""
This code is created based on
https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pascal_tf_record.py
and
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md

The structure of the dataset in PASCAL VOC format must be
- main_folder
    - images
        - ....jpeg
        - ....jpeg
        - ...
    - annotions
        - ....xml
        - ....xml
        - ...
- filename_list.txt

Convert raw PASCAL dataset to TFRecord for object_detection.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os

from lxml import etree
import PIL.Image
import tensorflow as tf

import os
import sys

# from object_detection.utils import dataset_util
# from object_detection.utils import label_map_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags

# directory name of dataset
flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.')

# name of directory with xml files for annotation
flags.DEFINE_string('annotations_dir', 'annotations',
                    '(Relative) path to annotations directory.')

# output file name (must end with .record)
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')

# path to the .pbtxt file which maps labels in string to integers
flags.DEFINE_string('label_map_path', 'label_map_sdc.pbtxt',
                    'Path to label map proto')

flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
FLAGS = flags.FLAGS


def dict_to_tf_example(data, dataset_directory,label_map_dict,
                        ignore_difficult_instances=False,
                        image_subdirectory='images'):

  """
  This functin is for converting xml derived dict to tf.Example prot.

  Notice that this function normalizes the bounding box coordinates
  provided by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding PASCAL dataset
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the
      PASCAL dataset directory holding the actual image data.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """

  # full_path: path to image from the directory to run python command
  img_path = os.path.join(image_subdirectory, data['filename'])
  full_path = os.path.join(dataset_directory, img_path)

  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)

  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')

  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  if 'object' in data:
    for obj in data['object']:
      difficult = bool(int(obj['difficult']))
      if ignore_difficult_instances and difficult:
        continue

      difficult_obj.append(int(difficult))

      xmin.append(float(obj['bndbox']['xmin']) / width)
      ymin.append(float(obj['bndbox']['ymin']) / height)
      xmax.append(float(obj['bndbox']['xmax']) / width)
      ymax.append(float(obj['bndbox']['ymax']) / height)
      classes_text.append(obj['name'].encode('utf8'))
      classes.append(label_map_dict[obj['name']])
      truncated.append(int(obj['truncated']))
      poses.append(obj['pose'].encode('utf8'))

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }))

  return example


def main(_):

    # data directory
    data_dir = FLAGS.data_dir

    # for output
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    # label map dictionary
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    # path to text file containing with list of jpeg file names (excluding .jpeg)
    examples_path = os.path.join(data_dir, 'filename_list.txt')
    examples_list = dataset_util.read_examples_list(examples_path)

    # path to folder containing annotations
    annotations_dir = os.path.join(data_dir, FLAGS.annotations_dir)

    for example in examples_list:

        # path to xml file
        path = os.path.join(annotations_dir, example + '.xml')

        # read xml file
        with tf.gfile.GFile(path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

        # to tf.Example format and write to output file
        tf_example = dict_to_tf_example(data, FLAGS.data_dir, label_map_dict,
                                                FLAGS.ignore_difficult_instances)
        writer.write(tf_example.SerializeToString())

    writer.close()

if __name__ == '__main__':
  tf.app.run()
