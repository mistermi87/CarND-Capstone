## Converting Pascal VOC format to TFRecord format

### Overview
This markdown explains how to convert Pascal VOC format data
into TFRecord format.

### Contents

This data folder contains the followings:

- `create_pascal_tf_record_sdc.py`: Python code for converting from Pascal VOC to TFRecord.
- `label_map_sdc.pbtxt`: This file is used for mapping class labels ("Green", "Red", "Yellow")
to integer ids (1, 2, 3).
- `bag2image.launch`: This file can be used for creating jpeg images from a ROS bag data.
- `train_val_generate.ipynb`: This Jupyter notebook can be used to spilt a dataset to
training and validation sets.
- `mp4_to_jpeg.ipynb`: This Jupyter notebook is for converting mp4 videos
to images and resizing them.
- `test_images`: Put test images here when running `test_performance.py` to
see how the trained model works. The results with bounding boxes are also stored
here after the run.

<!-- - `rosbag_data`: Folder containing all the ROS bag data4.
- `rosbag_data_train`: Folder for the training set created from the ROS bag data.
- `rosbag_data_train/images`: This folder contains images in jpeg format.
- `rosbag_data_train/annotations`: This folder contains xml files containing label info of images.
- `rosbag_data_train/filename_list.txt`: This text file contain a list of image file
names without the file extension `.jpeg` (or equivalently a list of xml filenames without the extension `.xml`)
- `rosbag_data_train/rosbag_data_train.record`: This TFRecord file is the output obtained by running `create_pascal_tf_record_sdc.py`.
- `rosbag_data_val`: Folder for the validation set created from the ROS bag data.
- `sim_train`: Folder for the validation set created from the simulator data.
- `sim_val`: Folder for the validation set created from the simulator data. -->

<!-- Note that the structure inside `rosbag_data_val`, `sim_train` or `sim_val`
is the same as that of `rosbag_data_train` folder. -->

The datasets for building traffic light detection models can be downloaded
from the links below:

- ROS bag dataset
  - [rosbag_data](https://drive.google.com/open?id=1sEXrzjLgBEjWGV2igexEtHGzDsaxFsdF): Full data from the ROS bag video.
  New training and validation sets can be created from this dataset by using `train_val_generate.ipynb`.  
  - [rosbag_data_train](https://drive.google.com/open?id=1yWSGbh7mgTuUCO5u4CZ-OMXFaEF1B4_w): 80% of randomly selected data from the full ROS bag data
  with `train_val_generate.ipynb`. For training.
  - [rosbag_data_val](https://drive.google.com/open?id=1AAQv7zRtcdPdQjiNqN3nhOvxo1c_xpcQ): The rest of data (20%). For validation.

- Simulator dataset (**UNDER CONSTRUCTION**)
  - [sim_data(training + validation)](???): Full data from recorded simulator videos.
  New training and validation sets can be created from this dataset by using `train_val_generate.ipynb`.  
  - [sim_data_train](???): 80% of randomly selected data from the full simulator data
  with `train_val_generate.ipynb`. For training.
  - [sim_data_val](???): The rest of data (20%). For validation.

The organization of the folder for each dataset is as follows:
- `images`: This subfolder contains images in jpeg format.
- `annotations`: This subfolder contains xml files with label info of images.
- `filename_list.txt`: This text file contain a list of image file
names without the file extension `.jpeg` (or equivalently a list of xml filenames without the extension `.xml`)
- `[dataset name].record`: This TFRecord file is obtained by running `create_pascal_tf_record_sdc.py`.


### Create Dataset in Pascal VOC Format.
In case one wants to create a new dataset in Pascal VOC format,
follow the instruction here.

#### Create Jpeg from ROS Bag Data
Follow the instruction below to create Jpeg images from the ROS bag file:

- The ROS bag file can be downloaded from [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip).
Unzip this file.

- For configuring RViz, download [rviz config file](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/default.rviz)
and put it on `~/.rviz/default.rviz`.

- After the above, follow the steps below to see the contents of the ROS bag file
on RViz:
  - On the first terminal, run `roscore`
  - On the second terminal, run `rosbag play -l [ROS bag filename]`
  - ON the third terminal, run `rviz`

- In case one wants to generate jpeg files from a ROS bag file, follow the steps below:
  - In `bag1image.launch`, change the path to the bag file and the one to the folder
  to save jpeg files.
  - On a terminal, run `roslaunch ./bag1image.launch`

#### Create Jpeg from Simulator Recording.

After recording some runs on the simulator, extract parts with traffic lights,
combine to form a single mp4 file. Then use `mp4_to_jpeg.ipynb` to create
Jpeg images.   

#### Create Xml File for Annotation.

To annotate images, use an annotation tool for creating Pascal VOC type data.
Here we use [LabelImg](https://github.com/tzutalin/labelImg).
A packaged app version for Mac is available [here](https://github.com/jiyeqian/labelImg/releases/download/1.4.4-pre/labelImg_qt5py3_mac_latest.zip). Note that the class names must be consistent with those given in
`label_map_sdc.pbtxt`.

#### Create Text File Listing File Names
To create `filename_list.txt`, use the following command
```
ls images | grep ".jpg" | sed s/.jpg// > filename_list.txt
```


### Converting to TFRecord

For using `create_pascal_tf_record_sdc.py` to create TFRecord format,
please follow the steps below (here we use the dataset in `rosbag_data_train`
for a concrete example):

1. Download the training+validation set from the link above,
unzip and place it to `models/research/tl_detect/data`.

2. Move to `models/research` and run
```
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

3. Go to the directory `models/research` and
run `create_pascal_tf_record_sdc.py` as follow:
```
PATH_TO_DATA_DIR=tl_detect/data
DATA_NAME=rosbag_data_train
python ${PATH_TO_DATA_DIR}/create_pascal_tf_record_sdc.py
--label_map_path=${PATH_TO_DATA_DIR}/label_map_sdc.pbtxt
--data_dir=${PATH_TO_DATA_DIR}/${DATA_NAME}
--output_path=${PATH_TO_DATA_DIR}/${DATA_NAME}/${DATA_NAME}.record
```

<!-- In case of using a new dataset, create a new dataset inside this
folder (`tl_detect/data`) such that it has
- `images` folder with jpeg images
- `annotations` folder with xml annotations.
- `filename_list.txt` listing in the name of the images (excluding the file extension). -->

In case of using a different dataset,
when running the `create_pascal_tf_record_sdc.py`, do not forget to
change `DATA_NAME=[new dataset folder]` appropriately.

### Note

- `create_pascal_tf_record_sdc.py` is created based on
  - `create_pascal_tf_record.py` available at  [Official site](https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pascal_tf_record.py)
  - [Officia instruction](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/preparing_inputs.md) on how to run `create_pascal_tf_record.py`
  - [Official instruction](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md) on personal TFRecord dataset creation
