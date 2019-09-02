from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import argparse
import time
import shutil
from decimal import Decimal
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt

tf.logging.set_verbosity(tf.logging.ERROR)
from densenet import DenseNet169
from pipeline import ImageReader, load_dataframes, get_body_part_dataframes, read_labeled_image_list, valid_transforms


DATA_DIRECTORY =  '/home/anicet/Datasets/'#'/scratch/hnkmah001/Datasets/'
BODY_PART = 'all'
RESTORE_FROM = '/home/anicet/tmp/snapshots/'#'/scratch/hnkmah001/densenet/snapshots/'
WEIGHTS_PATH   = '/home/anicet/Pretrained_models/densenet169.pkl'#'/scratch/hnkmah001/Pretrained_models/densenet169.pkl'
MODELS = {"ELBOW":'elbow_0/model.ckpt-11088', "FINGER":'finger_0/model.ckpt-12122', "FOREARM":'forearm_0/model.ckpt-3192',
            "HAND":'hand_0/model.ckpt-13840', "HUMERUS":'humerus_0/model.ckpt-2385', "SHOULDER":'shoulder_0/model.ckpt-14658',
            "WRIST":'wrist_0/model.ckpt-17066'}

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="densenet_169 Network for MURA")
    parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the MURA dataset.")
    parser.add_argument("--bpart", type=str, default=BODY_PART,
                        help="The body part to use for training")
    parser.add_argument("--restore_from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--weights_path", type=str, default=WEIGHTS_PATH,
                        help="Path to the file with pretrained weights. ")
    return parser.parse_args()


def load(loader, sess, ckpt_path):
    '''Load trained weights.

    Args:
      loader: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    loader.restore(sess, ckpt_path)
    print("\nRestored model parameters from {}".format(ckpt_path))


def main():
    """Create the model and start the training."""
    args = get_arguments()


    # Create queue coordinator.
    coord = tf.train.Coordinator()

    train, valid, valid_studies = load_dataframes(DATA_DIR = args.data_dir)
    _, _, valid_studies_df = get_body_part_dataframes(train, valid, valid_studies, args.bpart)

    valid_studies_df_list = read_labeled_image_list(valid_studies_df)
    valid_studies_path = valid_studies_df_list[0]
    valid_studies_label = valid_studies_df_list[1]
    number_of_validation_studies = len(valid_studies_df_list[1])

    print("\nNumber of validation studies for %s dataset:"%args.bpart, number_of_validation_studies)

    image = tf.placeholder(tf.float32, [None, 320, 320, 3])

    # Create network  with weights initialized from densenet_169 pretrained on ImageNet
    net = DenseNet169(args.weights_path)

    # Predictions
    prob = net.build(inputs=image, is_training=False)
    prob = tf.reshape(prob, [-1])

    all_variables = tf.all_variables()
    #config = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    # Load weights
    saver = tf.train.Saver(var_list=all_variables)
    #load(saver, sess, args.restore_from)

    # Start queue threads
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    probabilities = np.zeros(number_of_validation_studies)
    predictions =np.zeros(number_of_validation_studies, dtype=int)
    previous_study_type = ""

    for i in tqdm(range(number_of_validation_studies),  desc='Evaluation'):
        img_list = [f for f in os.listdir(valid_studies_path[i]) if not f.startswith(".")]
        num_img = len(img_list)
        pred_study = np.zeros(num_img)
        for j in range(num_img):
            img_path = valid_studies_path[i]+img_list[j] # eg. of path: 'MURA-v1.1/valid/XR_ELBOW/patient99999/study1_positive/image1.png'
            study_type = img_path.split("XR_")[1]  # Extract the study type in path between "XR_" and "/patient"
            study_type = study_type.split("/patient")[0]
            restore_from = args.restore_from+MODELS[study_type]
            if study_type != previous_study_type:
                load(saver, sess, restore_from)
            img_contents = tf.read_file(img_path)
            img = tf.image.decode_png(img_contents, channels=3)
            img = valid_transforms(img, study_type.lower()) # Normalize each model's inputs with the same statistics it has been trained on.
            img = tf.expand_dims(img, axis=0)
            img_arr = sess.run(img)
            feed_dict = {image: img_arr}
            pred_img = sess.run(prob, feed_dict=feed_dict)
            pred_study[j] = pred_img[0]
            previous_study_type = study_type
            #print('{:.4f}'.format(pred_study[j]))
        pred_study_mean = np.mean(pred_study)
        if pred_study_mean >0.5:
            predictions[i] = 1
        else:
            predictions[i] = 0
        probabilities[i] = pred_study_mean

    labels = tf.convert_to_tensor(valid_studies_label, dtype=tf.int32)
    predictions = tf.convert_to_tensor(predictions, dtype=tf.int32)
    probabilities = tf.convert_to_tensor(probabilities, dtype=tf.float32)

    # Define metrics
    confusion_matrix = tf.confusion_matrix(labels=labels, predictions=predictions)
    accuracy = tf.contrib.metrics.accuracy(labels=labels, predictions=predictions)
    auc, auc_update_op = tf.metrics.auc(labels=labels, predictions=probabilities)
    recall, recall_update_op = tf.metrics.recall(labels=labels, predictions=predictions)
    kappa, kappa_op = tf.contrib.metrics.cohen_kappa(labels=labels, predictions_idx=predictions, num_classes=2)

    #config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    kappa_val, kappa_op_val, probs, testy, confusion_matrix_val, accuracy_val, auc_val, auc_op, recall_val, recall_op = sess.run([kappa, kappa_op, probabilities, labels, confusion_matrix, accuracy, auc, auc_update_op, recall, recall_update_op])

    print('\nConfusion matrix:\n', confusion_matrix_val)
    print('\nArea under the ROC curve:', auc_op)
    print("\nRecall:", recall_op)
    print("\nAccuracy:", accuracy_val)
    print("\nCohen's kappa:", kappa_op_val)


    # Plot the roc curve
    fpr, tpr, thresholds = roc_curve(testy, probs)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(fpr, tpr, marker='.')
    plt.show()

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    main()
