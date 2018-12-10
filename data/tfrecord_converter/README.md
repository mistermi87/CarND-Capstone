## Converting Pascal VOC format to TFRecord format.

### Contents

- `create_pascal_tf_record_sdc.py`: Python code for converting from Pascal VOC to TFRecord.
- `SDC_ROS/annotations`: This folder contains xml files containing label info of images.
- `SDC_ROS/images`: This folder contains images in jpeg format.
- `SDC_ROS/filename_trainval.txt`: This text file contain a list of image file
names without the file extension `.jpeg` (or equivalently a list of xml file names without the extension `.xml`)
- `SDC_ROS/label_map_sdc.pbtxt`: This file is used for mapping class labels ("Green", "Red", "Yellow")
to integer ids (1, 2, 3).
- `SDC_ROS/sdc_ros.record`: This TFRecord file is the output obtained by running `create_pascal_tf_record_sdc.py`.

### Instruction

For using `create_pascal_tf_record_sdc.py` to create TRRecord format,
please follow the steps below:

1. Clone the original repo for tensorflow models:
```
git clone https://github.com/tensorflow/models
```
2. Follow the [original instruction](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) to install some dependencies and setup.

3. Move the folder `SDC_ROS` to `models/research`.

4. Move `create_pascal_tf_record_sdc.py` to
`models/research/python object_detection/dataset_tools/`.

5. Go to `models/research` and run `create_pascal_tf_record_sdc.py`
as follow:
```
cd models/research
python object_detection/dataset_tools/create_pascal_tf_record_sdc.py   
--label_map_path=SDC_ROS/label_map_sdc.pbtxt
--data_dir=SDC_ROS
--set=trainval
--output_path=SDC_ROS/sdc_ros.record
```

### Note

- To use `create_pascal_tf_record_sdc.py`, the data folder must contain
a folder named `annotations` with xml files, a folder named `images` with jpeg files,
`filename_[...].txt` file (where `[...]` is `train`, `val`, `test` or `trainval`,
depending on the purpose of this dataset), and `label_map_sdc.pbtxt`
file to map string labels to integer ids.

- To create `filename_[...].txt`, use the following command:
```
ls images | grep ".jpg" | sed s/.jpg// > filename_[...].txt
```
- In case of errors, here are some hints based on my experience:
  - After moving to `/models/research`, set path as
  ```
  export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
  ```
  - Update `protobuf` (Protocol Buffer by Google Developers).

- `create_pascal_tf_record_sdc.py` is created based on
  - `create_pascal_tf_record.py` available at  [Official site](https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pascal_tf_record.py)
  - [Officia instruction](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/preparing_inputs.md) on how to run `create_pascal_tf_record.py`
  - [Official instruction](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md) on personal TFRecord dataset creation

- A general explanation on the conversion to TFRecord and use of the converted
data to object detection:
  - [Racoon Detector](https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9)
  - [Peach Detector](https://medium.com/practical-deep-learning/a-complete-transfer-learning-toolchain-for-semantic-segmentation-3892d722b604)
  - [Dog Detector](http://androidkt.com/train-object-detection/)
