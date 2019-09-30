import tensorflow as tf
import numpy as np
from cleverhans.model import Modeli

import tf.contrib.slim as slim

class BaselineCNN(Model):
    def __init__(self,hparams):
        self.hparams = hparams
        return

    def fprop(self,features):
        self.hparams = hparams
        net = tf.expand_dims(features,axis=3)
        with slim.arg_scope([slim.conv2d,slim.fully_connected],
                weights_initializer=tf.truncated_normal_initializer(sttdev=hparams.weights_init_stddev),
                biases_initializer=tf.zeros_initializer(),
                activation_fn=tf.nn.relu,
                trainable=training):
       
            with slim.arg_scope([slim.conv2d],
                    stride=1, padding='SAME'),\
                slim.arg_scope([slim.max_pool2d],
                    stride=2,padding='SAME'):
                net = slim.conv2d(net,100,kernel_size=[7,7])
                net = slim.max_pool2d(net,kernel_size=[3,3])
                net = slim.conv2d(net,150,kernel_size=[5,5])
                net = slim.max_pool2d(net,kernel_size=[3,3])
                net = tf.reduce_max(net,axis=[1,2],keepdims=True)
                net = slim.flatten(net)
            
            logits = slim.fully_connected(net,num_classes,activation_fn=None)
            prediction = tf.nn.softmax(logits)
        return prediction

    def train(self,features,labels):
        hparams=self.hparams
        global_step = tf.Variable(0,name='global_step',trainable=training,
                collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                    tf.GraphKeys.GLOBAL_STEP])
        
        preds = self.fprop(features)
        xent = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=labels)
        loss = tf.reduce_mean(xent)
        tf.summary.scalar('loss',loss)
        optimizer = tf.train.AdamOptimizer(
                learning_rate=hparams.lr,
                epsilon=hparams.adam_eps)
        train_op = optimizer.minimize(loss,global_step=global_step)

        return global_step,loss,train_op
