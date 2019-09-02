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
tf.logging.set_verbosity(tf.logging.ERROR)
from densenet import DenseNet169
from pipeline import ImageReader, load_dataframes, get_body_part_dataframes, read_labeled_image_list


BATCH_SIZE = 8
DATA_DIRECTORY = '/home/anicet/Datasets/' #'/scratch/hnkmah001/Datasets/'
LEARNING_RATE = 1e-4
MOMENTUM = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 20
BODY_PART = 'all'
RESTORE_FROM = None
SNAPSHOT_DIR = './snapshots/'#'/scratch/hnkmah001/densenet/snapshots/'
WEIGHTS_PATH   = '/home/anicet/Pretrained_models/densenet169.pkl'#'/scratch/hnkmah001/Pretrained_models/densenet169.pkl'
SUMMARIES_DIR  = './summaries/'#'/scratch/hnkmah001/densenet/summaries/'



def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="densenet_169 Network for MURA")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the MURA dataset.")
    parser.add_argument("--bpart", type=str, default=BODY_PART,
                        help="The body part to use for training")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                        help="Learning rate for training.")
    #parser.add_argument("--momentum", type=float, default=MOMENTUM,
    #                    help="Momentum parameter")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY,
                        help="Weight decay parameter")
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS,
                        help="Number of training epochs.")
    parser.add_argument("--restore_from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weights_path", type=str, default=WEIGHTS_PATH,
                        help="Path to the file with pretrained weights. ")
    parser.add_argument("--summaries_dir", type=str, default= SUMMARIES_DIR,
                        help="Path to the file where variables are saved for TensorBoard.")
    return parser.parse_args()

def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')

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
    train_df, valid_df, valid_studies_df = get_body_part_dataframes(train, valid, valid_studies, args.bpart)

    train_df_list = read_labeled_image_list(train_df) # Returns a tuple (train_path_list, train_label_list)
    valid_df_list = read_labeled_image_list(valid_df)
    number_of_training_images = len(train_df_list[1]) # Numer of labels
    number_of_validation_images = len(valid_df_list[1])
    NUM_STEPS = args.num_epochs*number_of_training_images//args.batch_size
    VALIDATION_STEPS = 5 #number_of_validation_images #// args.batch_size
    EVALUATE_EVERY = 10 #number_of_training_images // args.batch_size # Evaluate every epoch
    A_train = sum(train_df_list[1])                     # Number of abnormals examples in the training dataset
    N_train = number_of_training_images - A_train       # Number of normal examples in the training dataset
    wT1 = N_train/(A_train+N_train)
    wT0 = A_train/(A_train+N_train)
    A_valid = sum(valid_df_list[1])                     # Number of abnormal examples in the validation dataset
    N_valid = number_of_validation_images - A_valid     # Number of normal examples in the validation dataset
    df = pd.DataFrame([[A_train, A_valid, wT1], [N_train, N_valid, wT0],[A_train+N_train, A_valid+N_valid, wT0+wT1]],
    index = ["Abnormal", "Normal", "Total"],
    columns = ["Train", "Valid", "Loss weights"]
     )
    print("\n%s dataset summary: \n "%args.bpart)
    print(df)
    print("\n")

    # Load reader.
    with tf.name_scope("Inputs"):
        reader = ImageReader(train_df, valid_df, args.bpart)
        image_batch, label_batch = reader.dequeue_train(args.batch_size)
        val_image_batch, val_label_batch = reader.dequeue_val(1)

    # Create network  with weights initialized from DenseNet169 pretrained on ImageNet
    net = DenseNet169(args.weights_path)

    # Define loss and accuracy
    loss = net.weighted_cross_entropy_loss(image_batch, label_batch, w0=wT0, w1=wT1, is_training=True, scope='train_loss')
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    acc = net.accuracy(image_batch, label_batch, is_training=True, scope='train_accuracy')

    # Define summaries for TensorBoard visualization
    loss_summary = tf.summary.scalar('training loss', loss)
    val_image_summary = tf.summary.image('validation input', val_image_batch)

    # Optimization ops
    learning_rate = tf.constant(args.learning_rate)
    optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate)
    trainable_variables = tf.trainable_variables()
    all_variables = tf.all_variables()
    with tf.control_dependencies(update_ops):
        optim = optimiser.minimize(loss, var_list=trainable_variables)

    # Track performance on the validation set during training
    val_loss = net.weighted_cross_entropy_loss(val_image_batch, val_label_batch, w0=wT0,
                                              w1=wT1, is_training=False, scope='Validation_loss')
    val_acc = net.accuracy(val_image_batch, val_label_batch, is_training=False, scope='Validation_accuracy')

    #config = tf.ConfigProto()
    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    if os.path.exists(args.summaries_dir+"%s"%args.bpart):
        shutil.rmtree(args.summaries_dir+"%s"%args.bpart)

    train_writer = tf.summary.FileWriter(args.summaries_dir+"%s"%args.bpart+"/train", sess.graph)
    val_writer = tf.summary.FileWriter(args.summaries_dir+"%s"%args.bpart+"/val")

    init_g = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    sess.run([init_g, init_l])

    # Saver for storing the last 40 checkpoints of the model.
    saver = tf.train.Saver(var_list=all_variables, max_to_keep=40)
    if args.restore_from is not None:
        load(saver, sess, args.restore_from)

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps.
    starting_time=time.asctime(time.localtime())

    for step in range(NUM_STEPS+1):
        start_time = time.time()

        if step % EVALUATE_EVERY == 0:
            # Calculate the validation loss and accuracy over the whole validation set
            val_loss_list = []
            val_acc_list = []

            for i in tqdm(range(VALIDATION_STEPS), desc="Validation"):
                val_image_summary_value, val_loss_i, val_acc_i = sess.run([val_image_summary, val_loss, val_acc])
                val_loss_list.append(val_loss_i)
                val_acc_list.append(val_acc_i)
                val_writer.add_summary(val_image_summary_value, step)
            val_loss_mean = np.mean(val_loss_list) # validation loss of the whole validation data
            val_acc_mean = np.mean(val_acc_list)

            # Reduce the learning rate if the valiatiation loss plateaus after one epoch
            if step > EVALUATE_EVERY:
                if val_loss_mean >= previous_val_loss:
                    learning_rate = tf.divide(learning_rate, 10.0)
                    print("Reducing the learning rate\n")
            previous_val_loss = val_loss_mean

            save(saver, sess, args.snapshot_dir+"%s"%args.bpart, step)

            summary=tf.Summary()
            summary.value.add(tag='validation loss', simple_value = val_loss_mean)
            summary.value.add(tag='validation accuracy', simple_value= val_acc_mean)
            val_writer.add_summary(summary, step)

            duration = time.time() - start_time
            print("\nSTEP {:d}/{:d} VALIDATION LOSS = {:.4f}, \t ACC = {:.4f},  \t ({:.3f} sec/step)".format(step, NUM_STEPS, val_loss_mean, val_acc_mean,  duration))
        else:
            loss_summary_value,  loss_value, acc_value, lr, _  = sess.run([loss_summary, loss, acc, learning_rate, optim])
            duration = time.time() - start_time
            train_writer.add_summary(loss_summary_value, step)
            print("step {:d}/{:d} \t loss = {:.4f}, \t acc = {:.4f},\t lr = {:.1e},  \t ({:.3f} sec/step)".format(step, NUM_STEPS, loss_value, acc_value, Decimal(lr.item()),  duration))

    end_time=time.asctime(time.localtime())

    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    main()
