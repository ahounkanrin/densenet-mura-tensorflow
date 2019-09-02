from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import pandas as pd
from tqdm import tqdm

#MEAN = [0.485, 0.456, 0.406] # ImageNet mean
#STD = [0.229, 0.224, 0.225] # ImageNet standard deviation

# Mean and std of MURA training set
mean_elbow = [8.505991, 8.505991, 8.505991]
std_elbow = [41.10112, 41.10112, 41.10112]
mean_finger = [8.214461, 8.214461, 8.214461]
std_finger = [43.545067, 43.545067, 43.545067]
mean_hand = [7.5668483, 7.5668483, 7.5668483]
std_hand = [33.53643, 33.53643, 33.53643]
mean_humerus = [32.02798, 32.02798, 32.02798]
std_humerus = [49.574905, 49.574905, 49.574905]
mean_forearm = [20.282866, 20.282866, 20.282866]
std_forearm = [41.685627, 41.685627, 41.685627]
mean_shoulder = [5.0057335, 5.0057335, 5.0057335]
std_shoulder = [35.797596, 35.797596, 35.797596]
mean_wrist = [4.300968, 4.300968, 4.300968]
std_wrist = [33.18204, 33.18204, 33.18204]
mean_all = [1.1395088, 1.1395088, 1.1395088]
std_all = [17.079645, 17.079645, 17.079645]

train_stats = {"all":[mean_all, std_all], "elbow":[mean_elbow, std_elbow],
    "finger":[mean_finger, std_finger], "hand":[mean_hand, std_hand],
    "humerus":[mean_humerus, std_humerus], "forearm":[mean_forearm, std_forearm],
    "shoulder":[mean_shoulder, std_shoulder], "wrist":[mean_wrist, std_wrist] }

BODY_PARTS= ["elbow", "finger", "forearm", "hand", "humerus", "shoulder", "wrist"]

def load_dataframes(DATA_DIR):
    """
     Import csv files into Dataframes.
    :return:
    """
    # import image path DataFrames
    train_df = pd.read_csv(
        os.path.join(DATA_DIR, "MURA-v1.1/train_image_paths.csv"),
        names=["path"]
    )
    valid_df = pd.read_csv(
        os.path.join(DATA_DIR, "MURA-v1.1/valid_image_paths.csv"),
        names=["path"]
    )

    valid_labeled_df = pd.read_csv(
        os.path.join(DATA_DIR, "MURA-v1.1/valid_labeled_studies.csv"),
        names=["path", "label"]
    )
    train_df.loc[train_df["path"].str.contains("positive"), "label"] = 1
    train_df.loc[train_df["path"].str.contains("negative"), "label"] = 0
    valid_df.loc[valid_df["path"].str.contains("positive"), "label"] = 1
    valid_df.loc[valid_df["path"].str.contains("negative"), "label"] = 0

    for part in BODY_PARTS:
        train_df.loc[train_df["path"].str.contains(part.upper()), "body_part"] = part
        valid_df.loc[valid_df["path"].str.contains(part.upper()), "body_part"] = part
        valid_labeled_df.loc[valid_labeled_df["path"].str.contains(part.upper()), "body_part"] = part

    # Replace the relative paths in the DataFrames by absolute paths
    train_df["path"] = DATA_DIR + train_df["path"].astype(str)
    valid_df["path"]  = DATA_DIR + valid_df["path"].astype(str)
    valid_labeled_df["path"]  = DATA_DIR + valid_labeled_df["path"].astype(str)

    # Convert float labels to integer
    train_df["label"] = train_df["label"].astype(int)
    valid_df["label"] = valid_df["label"].astype(int)
    valid_labeled_df["label"] = valid_labeled_df["label"].astype(int)
    return train_df, valid_df, valid_labeled_df  # train_labeled, valid_labeled,

def get_body_part_dataframes(train_df, valid_df, valid_labeled_df, bpart='all'):
    assert ((bpart in BODY_PARTS) or bpart=='all'), "Unrecognized body part selection: %s"%bpart
    if bpart == 'all':
        return train_df, valid_df, valid_labeled_df
    else:
        return train_df.loc[train_df["body_part"]==bpart], valid_df.loc[valid_df['body_part']==bpart], valid_labeled_df.loc[valid_labeled_df['body_part']==bpart]

def train_transforms(image, bpart):
    image = tf.expand_dims(image, axis=0) # Expand to 4-D shape required by tf.resize_nearest_neighbor
    image = tf.image.resize_nearest_neighbor(image, [320, 320])
    image = tf.squeeze(image)
    image = tf.cast(image, dtype=tf.float32)
    mean = train_stats[bpart][0]
    std = train_stats[bpart][1]
    image = image - mean
    image = image/std
    image = tf.image.random_flip_left_right(image)
    image = tf.contrib.image.rotate(image, angles=tf.random_uniform([], maxval=30.0))
    return image

def valid_transforms(image, bpart):
    image = tf.expand_dims(image, axis=0) # Expand to 4-D shape required by tf.resize_nearest_neighbor
    image = tf.image.resize_nearest_neighbor(image, [320, 320])
    image = tf.squeeze(image)
    image = tf.cast(image, dtype=tf.float32)
    mean = train_stats[bpart][0]
    std = train_stats[bpart][1]
    image = image - mean
    image = image/std
    return image

def read_labeled_image_list(df):
    """
    Args:
    df: dataframe with path to images
    Returns: Two lists with paths to images and corresponding labels.
    """
    images_list = df["path"].tolist()
    labels_list = df["label"].tolist()
    return images_list, labels_list

def read_image_from_disk(input_queue, bpart, is_training=True):
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_png(image_contents, channels=3)
    label = input_queue[1]
    if is_training:
        image = train_transforms(image, bpart)
    else:
        image = valid_transforms(image, bpart)
    return image, label

class ImageReader:
    """Image reader which reads images and corresponding label from disk, performs
     data augmentation, and enqueues into a TensorFlow queue.
    """

    def __init__(self, train_df, val_df, bpart):
        """
        Args:
            train_df: training dataframe
            val_df: validation DataFrame
            coord: TensorFlow queue coordinator
            is_training: whether it is the training phase or not
        """
        self.train_df = train_df
        self.val_df = val_df
        self.bpart = bpart

        self.image_list, self.label_list = read_labeled_image_list(self.train_df)
        self.images =  tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.int8)
        self.queue = tf.train.slice_input_producer([self.images, self.labels], shuffle=True)
        self.image, self.label = read_image_from_disk(self.queue, self.bpart, is_training=True)

        self.val_image_list, self.val_label_list = read_labeled_image_list(self.val_df)
        self.val_images =  tf.convert_to_tensor(self.val_image_list, dtype=tf.string)
        self.val_labels = tf.convert_to_tensor(self.val_label_list, dtype=tf.int8)
        self.val_queue = tf.train.slice_input_producer([self.val_images, self.val_labels], shuffle=False)
        self.val_image, self.val_label = read_image_from_disk(self.val_queue, self.bpart,  is_training=False)

    def dequeue_train(self, num_elements):
        """
        Pack training images and labels into a batch.
        """
        image_batch, label_batch = tf.train.batch([self.image, self.label], num_elements)
        return image_batch, label_batch

    def dequeue_val(self, num_elements):
        """
        Pack validation images and labels into a batch.
        """
        val_image_batch, val_label_batch = tf.train.batch([self.val_image, self.val_label], num_elements)
        return val_image_batch, val_label_batch
