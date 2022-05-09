#!/usr/bin/env python
#-*- encoding: utf8 -*-

import os
import sys
import rospy
import argparse
import numpy as np
import facenet.align.detect_face
import facenet.facenet as fn
import tensorflow.compat.v1 as tf
import math
from sklearn.svm import SVC
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main(argv):
    dataset_path = os.path.expanduser(argv.dataset_path)
    dataset_path = os.path.abspath(dataset_path)
    model_file = os.path.expanduser(argv.model)
    model_file = os.path.abspath(model_file)
    classifier_filename_exp = os.path.expanduser(argv.output_classifier)
    classifier_filename_exp = os.path.abspath(classifier_filename_exp)

    assert os.path.exists(dataset_path) is True, 'please correct path...'
    np.random.seed(seed=777)

    # TF
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    pnet, rnet, onet = facenet.align.detect_face.create_mtcnn(sess, None)

    fn.load_model(model_file)

    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    embedding_size = embeddings.get_shape()[1]

    dataset = fn.get_dataset(dataset_path)
    for cls in dataset:
        assert len(cls.image_paths) > 0, 'there must be at least one image for each class in the dataset'

    paths, labels = fn.get_image_paths_and_labels(dataset)
    print('Number of classes: %d' % len(dataset))
    print('Number of images: %d' % len(paths))

    nrof_images = len(paths)
    nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / 90))
    emb_array = np.zeros((nrof_images, embedding_size))

    for i in range(nrof_batches_per_epoch):
        start_index = i * 90
        end_index = min((i + 1) * 90, nrof_images)
        paths_batch = paths[start_index:end_index]
        images = fn.load_data(paths_batch, False, False, 160)
        feed_dict = {images_placeholder: images, phase_train_placeholder: False}
        emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

    model = SVC(kernel='linear', probability=True)
    model.fit(emb_array, labels)
    class_names = [cls.name.replace('_', ' ') for cls in dataset]

    with open(classifier_filename_exp, 'wb') as outfile:
        pickle.dump((model, class_names), outfile)
    print('Saved classifier model to file "%s"' % classifier_filename_exp)
    exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--output_classifier', type=str, required=True)

    main(parser.parse_args(sys.argv[1:]))
