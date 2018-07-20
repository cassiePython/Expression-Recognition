import tensorflow as tf
import functools
import math

# Dependency imports
import numpy as np

slim = tf.contrib.slim
    
def create_model(source_images=None, num_classes=None, dropout=None):
    
    ####################
    # Create model #
    ####################
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.random_normal_initializer(),
                      biases_initializer = tf.random_normal_initializer()
                      #weights_regularizer=slim.l2_regularizer(0.0005)):
                        ):
        with slim.arg_scope([slim.conv2d], stride=1, padding='VALID'):
            source_images = tf.reshape(source_images, shape=[-1, 64, 64, 1])
            net = slim.conv2d(source_images, 16, [5, 5], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            
            net = slim.conv2d(net, 8, [7, 7], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
                
            net = slim.conv2d(net, 16, [5, 5], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3') #4*4*16
            
            net = slim.flatten(net)
            net = slim.fully_connected(net, 100, scope='fc4')
            net = slim.dropout(net, dropout, scope='dropout4')
            net = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc5')
    return net

def create_loss(source_labels,pred_labels):

    ####################
    # Create loss #
    ####################
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=source_labels, logits=pred_labels))
    return cost
            
            
        
