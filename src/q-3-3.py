#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import tensorflow as tf
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,InputLayer


# In[ ]:


class KerasConvNet(tf.keras.Model):
    def __init__(self, dim1, dim2, n_classes):
        super().__init__()
        init_obj = tf.variance_scaling_initializer(scale=2.0)
        self.conv_layer_1 = tf.layers.Conv2D(dim1, [5,5], [1,1], padding='same',
                                      kernel_initializer=init_obj,
                                      activation=tf.nn.relu)
        self.conv_layer_2 = tf.layers.Conv2D(dim2, [3,3], [1,1], padding='same',
                                      kernel_initializer=init_obj,
                                      activation=tf.nn.relu)
        self.dense_layer = tf.layers.Dense(n_classes, kernel_initializer=init_obj)
        
        
    def call(self, X,training=None):
        out = None
        X = self.conv_layer_1(X)
        X = self.conv_layer_2(X)
        X = tf.layers.flatten(X)
        out = self.dense_layer(X)    
        return out


# In[ ]:


def train(model_init_fn, optimizer_init_fn, X_train, y_train):
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int32, [None,1])
    is_training = tf.placeholder(tf.bool, name='is_training')
        
    scores = model_init_fn(x, is_training)

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=scores)
    loss = tf.reduce_mean(loss)

    optimizer = optimizer_init_fn()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = optimizer.minimize(loss)
            

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        t = 0
        feed_dict = {x: X_train, y: y_train, is_training:1}
        loss_np, _ = sess.run([loss, train_op], feed_dict=feed_dict)
        print('Loss = %.4f' % (loss_np))


# In[ ]:


learning_rate = 3e-3
channel_1, channel_2, num_classes = 32, 16, 10

def model_init_fn(inputs, is_training):
    model = None
    input_shape = (32,32,3)
    channel_1, channel_2, num_classes = 32, 16, 10
    initializer = tf.variance_scaling_initializer(scale=2.0)
    layers = [
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Conv2D(channel_1, [5,5], [1,1], padding='same',
                               kernel_initializer=initializer,
                               activation=tf.nn.relu),
        tf.keras.layers.Conv2D(channel_2, [3,3], [1,1], padding='same',
                               kernel_initializer=initializer,
                               activation=tf.nn.relu),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, kernel_initializer=initializer)
    ]
    model = tf.keras.Sequential(layers)
    return model(inputs)

learning_rate = 5e-4
def optimizer_init_fn():
    optimizer = None
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, 
                                           use_nesterov=True)
    return optimizer

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
train(model_init_fn, optimizer_init_fn,X_train[:1000,:,:,:],y_train[:1000,:])


