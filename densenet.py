from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import pickle
import logging
import numpy as np

eps = 1e-20

class DenseNet169:

    def __init__(self, densenet169_pkl_path):

        with open(densenet169_pkl_path, 'rb') as f:
          self.data = pickle.load(f, encoding='latin1')
        print("\nPretrained weights loaded")
        self.data_dict={}
        for i in range(len(self.data)):
          key=self.data[i]['name']
          value=self.data[i]['weights']
          self.data_dict[key]=value

    def _kernel_initializer(self, name):
        """ Initialze weights from the pretrained network on ImageNet
        Args:
            name: Name of the layer in the pretrained network
        """

        filt = self.data_dict[name][0]   # Caffe format [output_depth, input_depth, filter_width, filter_height ]
        filt =  filt.transpose()         # convert to TF filter format [filter_height, filter_width, input_depth, output_depth]
        return tf.constant_initializer(value=filt, verify_shape=True)


    def unit(self, inputs, depth, conv_scope, caffeName, kernel, is_training, stride=1, rate=1, drop=0):
      """Basic unit. BN -> RELU -> CONV
      Args:
        inputs: A tensor of size [batch, height, width, channels].
        caffeName: Name of the layer in the pretrained network in Caffe
        is_training: Whether we are in training mode or not
        depth: The growth rate of the composite function layer.
               The num_outputs of bottleneck and transition layer.
        kernel: Kernel size.
        stride: The DenseNet unit's stride.
        rate: An integer, rate for atrous convolution.
        drop: The dropout rate of the DenseNet unit.
      """

      net = tf.layers.batch_normalization(inputs, training=is_training, name='BatchNorm')
      net = tf.nn.relu(net)
      net = tf.layers.conv2d(net, filters=depth, kernel_size=kernel, strides=stride,
        padding='same', dilation_rate=rate, activation=None, use_bias=False,
        kernel_initializer=self._kernel_initializer(caffeName), name=conv_scope)

      if drop > 0:
        net = tf.layers.dropout(net, rate=drop, training=is_training, name='dropout')
      return net

    def dense(self, inputs, growth, caffeName, is_training, bottleneck=True, stride=1,
              rate=1, drop=0, scope=None):
      """Dense layer.
      Args:
        inputs: A tensor of size [batch, height, width, channels].
        growth: The growth rate of the dense layer.
        caffeName: Name of the layer in the pretrained network in Caffe
        is_training: Whether we are in training mode or not
        bottleneck: Whether to use bottleneck.
        stride: The DenseNet unit's stride. Determines the amount of downsampling
        of the units output compared to its input.
        rate: An integer, rate for atrous convolution.
        drop: The dropout rate of the dense layer.
        scope: Optional variable_scope.
      """
      net = inputs
      if bottleneck:
        with tf.variable_scope('bottleneck'):
          caffeName_x1 = caffeName+'/x1'
          net = self.unit(net, is_training=is_training, depth=4*growth, conv_scope='conv1x1',
                     caffeName=caffeName_x1,  kernel=[1,1],stride=stride, rate=rate, drop=drop)

      with tf.variable_scope('composite'):
        caffeName_x2 = caffeName+'/x2'
        net = self.unit(net, is_training=is_training, depth=growth, conv_scope='conv3x3', caffeName=caffeName_x2,
                   kernel=[3,3], stride=stride, rate=rate, drop=drop)
      return net

    def transition(self, inputs, caffeName, is_training, bottleneck=True, compress=0.5, stride=1,
                   rate=1, drop=0, scope=None):
      """Transition layer.
      Args:
        inputs: A tensor of size [batch, height, width, channels].
        caffeName: Name of the layer in the pretrained network in Caffe
        is_training: Whether we are in training mode or not
        bottleneck: Whether to use bottleneck.
        compress: The compression ratio of the transition layer.
        stride: The transition layer's stride. Determines the amount of downsampling of the units output compared to its input.
        rate: An integer, rate for atrous convolution.
        drop: The dropout rate of the transition layer.
        scope: Optional variable_scope.
      Returns:
        The transition layer's output.
      """

      net = inputs
      if compress < 1:
        num_outputs = math.floor(inputs.get_shape().as_list()[3] * compress)
      else:
        num_outputs = inputs.get_shape().as_list()[3]

      net = self.unit(net, is_training=is_training, depth=num_outputs, conv_scope='conv1x1', caffeName=caffeName,
                 kernel=[1,1], stride=stride, rate=rate)
      net = tf.layers.average_pooling2d(net, pool_size=2, strides=2, padding='same', name='pool2x2')

      if drop > 0:
        net = tf.layers.dropout(net, rate=drop, training=is_training, name='dropout')
      return net

    def stack_dense_blocks(self, inputs, blocks, growth, is_training, bottleneck=True, compress=0.5,
      stride=1, rate=1, drop=0, scope=None):
      """Dense block.
      Args:
        inputs: A tensor of size [batch, height, width, channels].
        blocks: List of number of layers in each block.
        growth: The growth rate of the dense layer.
        bottleneck: Whether to use bottleneck.
        compress: The compression ratio of the transition layer.
        stride: The dense layer's stride. Determines the amount of downsampling of the units output compared to its input.
        rate: An integer, rate for atrous convolution.
        drop: The dropout rate of the transition layer.
        scope: Optional variable_scope.
      Returns:
        The dense block's output.
      """

      net = inputs
      for i, num_layer in enumerate(blocks):
        with tf.variable_scope('block%d' %(i+1), values=[net]):
          for j in range(num_layer):
            with tf.variable_scope('dense%d' %(j+1), values=[net]):
              identity = tf.identity(net)
              caffeName = 'conv%d_%d'%(i+2, j+1)
              dense_output= self.dense(net, growth, caffeName, is_training=is_training, bottleneck=bottleneck,
                                  stride=stride, rate=rate, drop=drop)
              net = tf.concat([identity, dense_output], axis=3, name='concat%d' %(j+1))

        if i < len(blocks) - 1:
          with tf.variable_scope('trans%d' %(i+1), values=[net]):
            caffeName = 'conv%d_blk'%(i+2)
            net = self.transition(net, caffeName, is_training=is_training, bottleneck=bottleneck, compress=compress,
                             stride=stride, rate=rate, drop=drop)
      return net

    def build(self, inputs, blocks=[6, 12, 32, 32], growth=32, bottleneck=True, compress=0.5, stride=1,
                 rate=1, drop=0, num_classes=1, is_training=True,reuse=tf.AUTO_REUSE, scope="DenseNet"):

      """Generator for DenseNet models.
      Args:
        inputs: A tensor of size [batch, height_in, width_in, channels].
        blocks: A list of length equal to the number of DenseNet blocks.
        growth: The growth rate of the DenseNet unit.
        bottleneck: Whether to use bottleneck.
        compress: The compression ratio of the transition layer.
        stride: The dense layer's stride. Determines the amount of downsampling of the units output compared to its input.
        drop: The dropout rate of the transition layer.
        num_classes: Number of predicted classes for classification tasks.
        is_training: Whether batch_norm and drop_out layers are in training mode.
        data_name: Which type of model to use.
        reuse: whether or not the network and its variables should be reused. To be
          able to reuse 'scope' must be given.
        scope: Optional variable_scope.
      Returns:
        net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      """
      with tf.variable_scope(scope, 'DenseNet', reuse=reuse):
          net = inputs
          net = tf.layers.conv2d(net, filters=growth*2, kernel_size=7, strides=2,
              padding='same', use_bias=False, activation=None,
              kernel_initializer=self._kernel_initializer('conv1'), name='conv7x7')
          net = tf.layers.batch_normalization(net, training=is_training, name='BatchNorm1')
          net = tf.nn.relu(net)
          net = tf.layers.max_pooling2d(net, pool_size=3, strides=2, padding='same', name='pool3x3')

          net = self.stack_dense_blocks(net, blocks, growth, is_training=is_training,
              bottleneck=bottleneck, compress=compress, stride=stride, rate=rate, drop=drop)
          net = tf.layers.batch_normalization(net, training=is_training, name='BatchNorm2')
          net = tf.nn.relu(net)

          # Global Average Pooling
          net = tf.reduce_mean(net, [1, 2], name='GAP', keepdims=True)

          # Fully connected layer
          net = tf.layers.dense(net, units=num_classes, activation=None, name='logits')
          net = tf.nn.sigmoid(net)
          return net

    def weighted_cross_entropy_loss(self, image_batch, label_batch, w0, w1, is_training, scope='None'):
        probabilities = self.build(inputs=image_batch, is_training=is_training)
        with tf.variable_scope(scope):
            probabilities = tf.reshape(probabilities, [-1])
            labels = tf.reshape(label_batch, [-1])
            loss_w1 = tf.multiply(w1, tf.cast(labels, dtype=tf.float32))*tf.log(probabilities+eps)
            loss_w0 = tf.multiply(w0, tf.cast((1-labels), dtype=tf.float32))*tf.log(1.0-probabilities+eps)
            loss = loss_w1+loss_w0
            loss = tf.multiply(loss, -1.0)
            loss = tf.reduce_mean(loss) # Take the mean of the batch loss
        return loss

    def accuracy(self, image_batch, label_batch, is_training, scope='None'):
        probabilities = self.build(inputs=image_batch, is_training=is_training)
        with tf.variable_scope(scope):
            probabilities = tf.reshape(probabilities, [-1])
            labels = tf.reshape(label_batch, [-1])
            labels = label_batch
            preds_neg = tf.zeros_like(labels)
            preds_pos = tf.ones_like(labels)
            predictions = tf.where(tf.math.greater(probabilities, 0.5), preds_pos, preds_neg)
            accuracy = tf.contrib.metrics.accuracy(labels, predictions)
        return accuracy
