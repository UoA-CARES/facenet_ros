# facenet_ros

This package is for face detecting and recognition using [facenet] in ROS. The code is heavily inspired by the [davidsandbergfacenet]

## Requirements

OpenCV, TensorFlow, etc.

        $ sudo pip install -r requirement.txt

Package that handles images E.g 
http://wiki.ros.org/cv_camera

## Install

Please note that all files in /data may need to be downloaded individually from github. This is due to the way github handles files. If the files are in ~kb then they have not been downloaded properly.

        $ git https://github.com/UoA-CARES/facenet_ros
        $ catkin_make or catkin build

## Usage

### Take image for registration

First you should make dataset directory any you want. For example, ~/dataset.

Image topic name for cv_camera /cv_camera/image_raw

Node that the image handling node needs to be running as well. For cv_camera this is $ rosrun cv_camera cv_camera_node

        $ rosrun facenet_ros take_image.py --image <rgb_image_topic_name> --destination ~/dataset/<Class Name>

<Class Name> is registered name, for example, John_Dow. After launch take_image node, look at the camera then the images are saved for every 1 second.

### Train classification

        $ rosrun facenet_ros train.py --dataset_path <dataset_directory> --model model/<Model-File> --output_classifier model/<classification_file>

### Run with camera

This package needs now only runs with an image and does not determinate face position

Image topic name for cv_camera /cv_camera/image_raw

Node that the image handling node needs to be running as well. For cv_camera this is $ rosrun cv_camera cv_camera_node

        $ rosrun facenet_ros node.py \_model_file:=<path_to_model_file> \_classifier_file:=<path_to_classification_file> image_raw:=<rgb_image_topic_name>


[facenet]: https://github.com/davidsandberg/facenet
[davidsandbergfacenet]: https://github.com/davidsandberg/facenet
