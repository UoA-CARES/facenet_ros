# face_recognition_facenet

This package is for face detecting and recognition using [facenet] in ROS. The code is heavily inspired by the [davidsandbergfacenet]

## Requirements

OpenCV, TensorFlow, etc.

        $ sudo pip install -r requirement.txt

## Install

        $ git clone https://stash.auckland.ac.nz/scm/~bahn915/face_recognition_facenet.git
        $ catkin_make or catkin build

## Usage

### Take image for registration

First you should make dataset directory any you want. For example, ~/dataset.

        $ rosrun face_recognition_facenet take_image.py --image <rgb_image_topic_name> --destination ~/dataset/<Class Name>

<Class Name> is registered name, for example, John_Dow. After launch take_image node, look at the camera then the images are saved for every 1 second.

### Train classification

        $ rosrun face_recognition_facenet train.py --dataset_path <dataset_directory> --model model/20170512-110547.pb --output_classifier <classification_file>

### Run with camera

This package needs pointcloud and rgb image.

        $ rosrun face_recognition_facenet node.py \_model_file:=./data/model/20170512-110547.pb \_classifier_file:=<classification_file> image_raw:=<rgb_image_topic_name>


[facenet]: https://github.com/davidsandberg/facenet
[davidsandbergfacenet]: https://github.com/davidsandberg/facenet
