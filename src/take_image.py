#!/usr/bin/env python
#-*- encoding: utf8 -*-

import os
import rospy
import argparse
import sys
import numpy as np
import tensorflow as tf
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import facenet.align.detect_face


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
minsize = 50  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor


class TakeImage:
    def __init__(self, args):
        rospy.init_node('take_photo', anonymous=False)

        destination_path = os.path.expanduser(args.destination)
        self.destination_path = os.path.abspath(destination_path)

        if not os.path.exists(self.destination_path):
            os.mkdir(self.destination_path)

        np.random.seed(seed=777)
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        self.pnet, self.rnet, self.onet = facenet.align.detect_face.create_mtcnn(self.sess, None)

        self.bridge = CvBridge()
        self.image_index = 0
        self.image_frame = 0
        self.image_frame_skip = 30

        rospy.Subscriber(args.image, Image, self.callback_image)
        rospy.loginfo('initialized...')
        rospy.spin()

    def callback_image(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        bounding_boxes, _ = facenet.align.detect_face.detect_face(cv_image, minsize, self.pnet, self.rnet, self.onet, threshold, factor)
        if bounding_boxes.shape[0] > 1:
            rospy.logwarn('there are many faces. just only one faces required...')
            return

        cropped = None
        for (x1, y1, x2, y2, acc) in bounding_boxes:
            w = x2 - x1
            h = y2 - y1
            #cv2.rectangle(cv_image, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (255, 0, 0), 2)

            bb = [0] * 4
            bb[0] = np.maximum(x1 - 44, 0)
            bb[1] = np.maximum(y1 - 44, 0)
            cropped = cv_image[int(bb[1]):int(bb[1] + h + 120), int(bb[0]):int(bb[0] + w + 88)]

        if cropped is not None:
            cv2.imshow("Image window2", cropped)
            self.image_frame += 1
            if self.image_frame > self.image_frame_skip:
                file_name = self.destination_path + '/image_' + str(self.image_index) + '.jpg'
                cv2.imwrite(file_name, cropped)
                self.image_frame = 0
                self.image_index += 1

        cv2.imshow("Image window", cv_image)
        cv2.waitKey(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    m = TakeImage(parser.parse_args(sys.argv[1:]))
