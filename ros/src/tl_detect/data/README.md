## Converting Pascal VOC format to TFRecord format

### Overview
This markdown explains how to convert Pascal VOC format data
into TFRecord format.

### Contents

This data folder contains the followings:

- `create_pascal_tf_record_sdc.py`: Python code for converting from Pascal VOC to TFRecord.
- `create_pascal_tf_record_sdc.bosch_mini.py`: Python code for converting from Pascal VOC to TFRecord for Bosch mini dataset.
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
- `boschmini_to_pascal.py`: This python file is to convert the original
yaml annotation file of the Bosch Small Traffic Lights Dataset to xml.
- `Readme-boschmini_to_pascal.md`: This markdown file explains
how to create Bosch mini dataset from the original Bosch Small Traffic Lights Dataset.

The datasets for building traffic light detection models can be downloaded
from the links given in `ros/src/tl_detect/README.md`

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
  - In `bag2image.launch`, change the path to the bag file and the one to the folder
  to save jpeg files.
  - On a terminal, run `roslaunch ./bag2image.launch`

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

In case of using a different dataset,
when running the `create_pascal_tf_record_sdc.py`, do not forget to
change `DATA_NAME=[new dataset folder]` appropriately.

### Creating Bosch Mini Dataset
For how to create Bosch mini dataset, please refer to
`Readme-boschmini_to_pascal.md` in this directory. 

### Note

- `create_pascal_tf_record_sdc.py` is created based on
  - `create_pascal_tf_record.py` available at  [Official site](https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pascal_tf_record.py)
  - [Officia instruction](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/preparing_inputs.md) on how to run `create_pascal_tf_record.py`
  - [Official instruction](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md) on personal TFRecord dataset creation
