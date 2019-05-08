#!/usr/bin/env python
# coding: utf-8


import os
import tensorflow as tf
import numpy as np


def convnet(X, conv_layer1_W, conv_layer2_W, conv_layer1_b, conv_layer2_b, dense_layer_W, dense_layer_b):

    scores = None
    conv_layer1_out = tf.nn.conv2d(X, conv_layer1_W, strides=[1,1,1,1], padding="SAME")
    conv_layer1_out += conv_layer1_b
    relu_layer1_out = tf.nn.relu(conv_layer1_out)
    
    conv_layer2_out = tf.nn.conv2d(conv_layer1_out, conv_layer2_W, strides=[1,1,1,1], padding="SAME")
    conv_layer2_out += conv_layer2_b
    relu_layer2_out = tf.nn.relu(conv_layer2_out)
    
    #flatten the output from ReLU layer
    N = tf.shape(relu_layer2_out)[0]
    relu_layer2_out = tf.reshape(relu_layer2_out, (N, -1))
    output = tf.matmul(relu_layer2_out, dense_layer_W)
    output += dense_layer_b
    
    return output


X = tf.placeholder(tf.float32)
conv_layer1_W = tf.random.normal((5, 5, 3, 6))
conv_layer2_W = tf.random.normal((2, 2, 6, 9))
conv_layer1_b = tf.random.normal((6,))
conv_layer2_b = tf.random.normal((9,))
dense_layer_W = tf.random.normal((32 * 32 * 9, 10))
dense_layer_b = tf.random.normal((10,))
scores = convnet(X, conv_layer1_W, conv_layer2_W, conv_layer1_b, conv_layer2_b, dense_layer_W, dense_layer_b)

X_np = np.random.rand(10, 32, 32, 3)
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    scores_np = sess.run(scores, feed_dict={X: X_np})
    print(scores_np)


