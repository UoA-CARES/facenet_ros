#!/usr/bin/env python
#-*- encoding: utf8 -*-

import os
import rospy
import cv2
import math
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PointStamped
from face_recognition_facenet.msg import RecognizedResult
from cv_bridge import CvBridge, CvBridgeError
import message_filters

import facenet.align.detect_face
import facenet.facenet as fn
import tensorflow as tf
import numpy as np
import pickle
from sklearn.svm import SVC
from scipy import misc
import itertools
import sensor_msgs.point_cloud2 as pc2


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
minsize = 40  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor


class FaceRecognitionFacenet:
    def __init__(self):
        np.random.seed(seed=777)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        self.pnet, self.rnet, self.onet = facenet.align.detect_face.create_mtcnn(self.sess, None)
        try:
            model_file = rospy.get_param('~model_file')
            classifier_file = rospy.get_param('~classifier_file')
        except KeyError:
            rospy.logerr('set ~model_file and ~classifier_file.')
            exit(-1)

        fn.load_model(model_file)

        self.classifier_filename_exp = os.path.expanduser(classifier_file)
        self.classifier_filename_exp = os.path.abspath(self.classifier_filename_exp)
        if not os.path.exists(self.classifier_filename_exp):
            self.is_have_classifier = False
            rospy.logwarn('train first.')
        else:
            self.is_have_classifier = True
            with open(self.classifier_filename_exp, 'rb') as infile:
                (self.model, self.class_names) = pickle.load(infile)

        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        self.embedding_size = self.embeddings.get_shape()[1]

        self.bridge = CvBridge()
        rgb_sub = message_filters.Subscriber('image_raw', Image)
        points_sub = message_filters.Subscriber('points', PointCloud2)
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, points_sub], 10, 0.5, allow_headerless=True)
        ts.registerCallback(self.callback_image)

        self.pub_debug_image = rospy.Publisher('result_image', Image, queue_size=10)
        self.pub_result = rospy.Publisher('recognized_faces', RecognizedResult, queue_size=10)
        rospy.loginfo('initialized...')

    def callback_image(self, img, points):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        if not self.is_have_classifier:
            bounding_boxes, _ = facenet.align.detect_face.detect_face(cv_image, minsize, self.pnet, self.rnet, self.onet, threshold, factor)
            for (x1, y1, x2, y2, acc) in bounding_boxes:
                w = x2 - x1
                h = y2 - y1
                cv2.rectangle(cv_image, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (255, 0, 0), 2)
        else:
            img_size = np.asarray(cv_image.shape)[0:2]
            bounding_boxes, _ = facenet.align.detect_face.detect_face(cv_image, minsize, self.pnet, self.rnet, self.onet, threshold, factor)
            nrof_faces = bounding_boxes.shape[0]

            if nrof_faces == 0:
                result_msg = RecognizedResult()
                self.pub_result.publish(result_msg)

                try:
                    self.pub_debug_image.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
                except CvBridgeError as e:
                    print(e)
                return

            img_list = [None] * nrof_faces
            aligned = None
            for i in xrange(nrof_faces):
                det = np.squeeze(bounding_boxes[i, 0:4])
                bb = np.zeros(4, dtype=np.int32)

                bb[0] = np.maximum(det[0] - 44 / 2, 0)
                bb[1] = np.maximum(det[1] - 44 / 2, 0)
                bb[2] = np.minimum(det[2] + 44 / 2, img_size[1])
                bb[3] = np.minimum(det[3] + 44 / 2, img_size[0])

                cropped = cv_image[bb[1]:bb[3], bb[0]:bb[2], :]
                aligned = misc.imresize(cropped, (160, 160), interp='bilinear')
                prewhitened = fn.prewhiten(aligned)
                img_list[i] = prewhitened

            images = np.stack(img_list)

            feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
            emb = self.sess.run(self.embeddings, feed_dict=feed_dict)

            predictions = self.model.predict_proba(emb)
            best_class_indeces = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indeces)), best_class_indeces]

            result_msg = RecognizedResult()
            index = 0
            for face in bounding_boxes:
                face = face.astype(int)
                cv2.rectangle(cv_image, (face[0], face[1]), (face[2], face[3]), (0, 255, 0), 2)

                if best_class_probabilities[index] >= 0.5:
                    cv2.putText(cv_image, self.class_names[best_class_indeces[index]], (face[0], face[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                    result_msg.name.append(self.class_names[best_class_indeces[index]])
                else:
                    cv2.putText(cv_image, 'unknown%d' % index, (face[0], face[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                    result_msg.name.append('unknown%d' % index)
                result_msg.confidence.append(best_class_probabilities[index])

                margin = int((face[2] - face[0]) * 0.3)
                x1 = face[0] + margin
                x2 = face[2] - margin
                y1 = face[1] + margin
                y2 = face[3] - margin

                req_pt = []
                for y in range(y1, y2):
                    for x in range(x1, x2):
                        req_pt.append([x, y])

                depth_point = pc2.read_points(points, skip_nans=False, field_names=('x', 'y', 'z'), uvs=req_pt)

                pp = np.empty((0, 3))
                for p in depth_point:
                    if not math.isnan(p[0]):
                        pp = np.append(pp, [[p[0], p[1], p[2]]], axis=0)

                pos = np.mean(pp, axis=0)
                ps1 = PointStamped()
                ps1.header.stamp = rospy.Time.now()
                ps1.header.frame_id = points.header.frame_id
                ps1.point.x = pos[0]
                ps1.point.y = pos[1]
                ps1.point.z = pos[2]

                result_msg.position.append(ps1)
                index += 1

            self.pub_result.publish(result_msg)

        try:
            self.pub_debug_image.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
            print(e)


if __name__ == '__main__':
    rospy.init_node("face_recognition_facenet", anonymous=False)
    m = FaceRecognitionFacenet()
    rospy.spin()
