## Building Traffic Light Detector

This markdown explains how to setup and train an object detection model
by using TensorFlow Object Detection API.

### Memo
- **Dec 13, 2018 (Tatsuo)**: Updated so that the codes works
with Tensorflow 1.4.0 and the created model runs with
Tensorflow 1.3.0 (and without `matplotlib`) on
`object_detection_sdc.ipynb` and `test_performance.py`.

### Contents

The contents of the folder `tl_detect` is as follows:

- `data`: This folder contains datasets for training models and
some files related to them (see `README.md` inside for further detail).
- `environment.yaml`: To create Anaconda environment to work to train a model
outside the workspace (Tensorflow 1.4.0).
- `object_detection_sdc.ipynb`: This Jupyter notebook is for checking the
performance of trained models. This is based on
`models/research/object_detection/object_detection_tutorial.ipynb`
provided in the [original repo](https://github.com/tensorflow/models) for Tensorflow models
as well as Object Detection Lab of this course.
- `object_detection_sdc_extra_lib.ipynb`: This one is essentially the same
as `object_detection_sdc.ipynb` but requires extra libraries such as
`matplotlib` and something from `models/research/object_detection/utlis` folder
in the [original repo](https://github.com/tensorflow/models) for Tensorflow models.
- `test_performance.py`: The content of this Python file is the same as
that in `object_detection_sdc.ipynb`.
- `training`: This folder is for training models.
- `training/model1`: This folder is for training models by starting with
the pre-trained model `faster_rcnn_inception_v2_coco_2017_11_08.tar.gz`
available at the [object detection model zoo](https://github.com/tensorflow/models/blob/69e1c50433c6cf7843a7cd337558efbb85656f07/research/object_detection/g3doc/detection_model_zoo.md).


### Instruction

#### For Training Outside Udacity Workspace

When working outside the Udacity Workspace for training,
create an environment with Anaconda:
```
conda env create -f environment.yaml
```
and activate
```
source activate carnd-capstone
```

In case one wants to use on Jupyter notebook, use
```
ipython kernel install --user --name carnd-capstone
```
and choose this kernel when using Jupyter notebook.

**NOTE THAT THIS ENVIRONMENT IS ONLY FOR TRAINING A MODEL**.
Some libraries are different versions than those in `requirements.txt`
of [CarND-Capstone repository](https://github.com/udacity/CarND-Capstone).
Some are not included there.
Please check if the performance of the trained model is
fine on the Udacity Workspace. Also note that the libraries
included in Udacity Workspace with CPU only are different
from those in the Workspace with CPU+GPU. In the end,
the check the performance of the trained model (and, before that if
it works for the object detection or not) on the Udacity Workspace.

#### Setup
The setup before training the model is as follows:

1. Clone the original repo for TensorFlow models:
```
git clone https://github.com/tensorflow/models
```

2. Select an old version (Tensorflow 1.4.0 but the generated file is compatible with
with Tensorflow 1.3.0) by using the following command:
```
cd models
git checkout 69e1c50
```
The website of this version is available
[here](https://github.com/tensorflow/models/tree/69e1c50433c6cf7843a7cd337558efbb85656f07).
This version is released in the end of 2017.

3. Follow the [original instruction](https://github.com/tensorflow/models/blob/69e1c50433c6cf7843a7cd337558efbb85656f07/research/object_detection/g3doc/installation.md) to install some dependencies and setup.

4. Put the folder `tl_detect` inside `models/research`

5. Download a pre-trained model and its configuration file from below
(In the following we assume that we have downloaded the pre-trained model file
`faster_rcnn_inception_v2_coco_2017_11_08.tar.gz` and
the configuration file `faster_rcnn_resnet101_coco.config`):
  - [pretrained models](https://github.com/tensorflow/models/blob/69e1c50433c6cf7843a7cd337558efbb85656f07/research/object_detection/g3doc/detection_model_zoo.md)
  - [configuration file](https://github.com/tensorflow/models/tree/69e1c50433c6cf7843a7cd337558efbb85656f07/research/object_detection/samples/configs)

5. Create a new folder (here we call `model1`) at `tl_detect/training`
and move `faster_rcnn_resnet101_coco.config` to `tl_detect/training/model1`.
Also create an empty folder with name `trained_model` there.

6. Unzip the tar.gz file and move the generated folder
(`faster_rcnn_inception_v2_coco_2017_11_08` here) to `tl_detect/training/model1`.

7. Edit the configuration file `faster_rcnn_inception_v2_coco.config`.
The following items must be edited (assuming that the ROS bag data is used):
  - `num_classes`: set to `3` so that it matches with the numbers of classes
  in `data/label_map_sdc.pbtxt`.
  - `fine_tune_checkpoint`: set to  `"tl_detect/training/model1/faster_rcnn_inception_v2_coco_2017_11_08/model.ckpt"`
  - `input_path` inside `train_input_reader`: set to the path to TFRecord file for training i.e. ``"tl_detect/data/rosbag_data_train/rosbag_data_train.record"``
  - `label_map_path` inside `train_input_reader`: set to the path to `label_map_sdc.pbtxt`, i.e.
  ``"tl_detect/data/label_map_sdc.pbtxt"``
  - `input_path` inside `eval_input_reader`: set to the path to TFRecord file for training, i.e. ``"tl_detect/data/rosbag_data_val/rosbag_data_val.record"``
  - `label_map_path` inside `eval_input_reader`: set to the path to `label_map_sdc.pbtxt`, i.e.
  ``"tl_detect/data/label_map_sdc.pbtxt"``
For the rest, change depending on how you train.

#### Training
For training, take the following steps:

1. Move to `models/research` and run
```
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

2. Command for training
```
MODEL_NAME=model1
CONFIG_FILE=faster_rcnn_inception_v2_coco.config
PATH_TO_MODEL_DIR=tl_detect/training/${MODEL_NAME}
python object_detection/train.py
--logtostderr
--train_dir=${PATH_TO_MODEL_DIR}
--pipeline_config_path=${PATH_TO_MODEL_DIR}/${CONFIG_FILE}
```

<!-- 3. Open a new terminal, move to `models/research` and run
(`MODEL_NAME`, `CONFIG_FILE` and `PATH_TO_MODEL_DIR` must be
the same as the previous step)
```
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
MODEL_NAME=model1
CONFIG_FILE=faster_rcnn_inception_v2_coco.config
PATH_TO_MODEL_DIR=tl_detect/training/${MODEL_NAME}
tensorboard --logdir='${PATH_TO_MODEL_DIR}'
```
Then access to the instructed page and monitor
the training process. -->

3. Once training is done, select one of the model check points
stored in `${PATH_TO_MODEL_DIR}` (here we take `model.ckpt-10`)
and use the following command to
create a graph file for the trained model.
```
CHECK_POINT=model.ckpt-10
python object_detection/export_inference_graph.py
--input_type=image_tensor
--pipeline_config_path=${PATH_TO_MODEL_DIR}/${CONFIG_FILE}
--trained_checkpoint_prefix=${PATH_TO_MODEL_DIR}/${CHECK_POINT}
--output_directory=${PATH_TO_MODEL_DIR}/trained_model/
```
A trained model is saved in `${PATH_TO_MODEL_DIR}/trained_model`.


4. To see the performance of the trained model,
open `object_detection_sdc.ipynb` at `models/research/tl_detect/`,
```
jupyter notebook object_detection_sdc.ipynb
```
edit `MODEL_NAME` (for the current case, set `MODEL_NAME='training/model1/trained_model'`)
and execute the cells. The test images are stored in `tl_detect/data/test_images`.
If you want to use more test images, add extra images to this folder
and edit `PATH_TO_TEST_IMAGES_DIR` and `TEST_IMAGE_PATHS` appropriately
in the Jupyter notebook.

In case one want to do the same thing without using Jupyter notebook,
run `test_performance.py` at  `models/research/tl_detect/` instead
(the code is exactly the same):
```
python `test_performance.py`
```
This `test_perfomrance.py` is designed such that it can run
on Udacity Workspace with and without GPU.


Depending on which pre-trained model and dataset to use,
change the paths in the `.config` file and `MODEL_NAME`, `CONFIG_FILE`
and `PATH_TO_MODEL_DIR`.


### Note

- In case of errors, here are some hints based on my experience:
  - After moving to `/models/research`, set path as
  ```
  export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
  ```
  <!-- - Update `protobuf` (Protocol Buffer by Google Developers). -->

- In the GPU environment of Udacity, `protobuf` is not installed
(while it is installed in CPU environment...). So, it is impossible
to train with Object Detection API there. But to run a trained
model seems to be possible (one can run `test_performance.py` there).
Also, `matplotlib` is not installed there...

<!-- - The version of TensorFlow must be `tensorflow>=1.9.0`. -->

- A general explanation on the conversion to TFRecord and use of the converted
data to object detection:
  - [Racoon Detector](https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9)
  - [Peach Detector](https://medium.com/practical-deep-learning/a-complete-transfer-learning-toolchain-for-semantic-segmentation-3892d722b604)
  - [Dog Detector](http://androidkt.com/train-object-detection/)

- [Data augmentation with configuration file](https://github.com/tensorflow/models/blob/a4944a57ad2811e1f6a7a87589a9fc8a776e8d3c/object_detection/builders/preprocessor_builder.py)

- [Official Object Detection API Tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html)
