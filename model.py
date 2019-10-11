import tensorflow as tf
import numpy as np
from cleverhans.model import Model

import tensorflow.contrib.slim as slim

class BaselineCNN(Model):
    def __init__(self,hparams,num_classes):
        self.hparams = hparams
        self.num_classes = num_classes
        return

    def fprop(self,features):
        hparams = self.hparams
        net = tf.expand_dims(features,axis=3)
        with slim.arg_scope([slim.conv2d,slim.fully_connected],
                weights_initializer=tf.truncated_normal_initializer(stddev=hparams.weights_init_stddev),
                biases_initializer=tf.zeros_initializer(),
                activation_fn=tf.nn.relu,
                trainable=True):
       
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
            
            logits = slim.fully_connected(net,self.num_classes,activation_fn=None)
            prediction = tf.nn.softmax(logits)
        return {self.O_LOGITS: logits,
                self.O_PROBS: prediction}

    def train(self,features,labels):
        hparams=self.hparams
        global_step = tf.Variable(0,name='global_step',trainable=True,
                collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                    tf.GraphKeys.GLOBAL_STEP])
        
        logits = self.get_logits(features)
        xent = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=labels)
        loss = tf.reduce_mean(xent)
        tf.summary.scalar('loss',loss)
        optimizer = tf.train.AdamOptimizer(
                learning_rate=hparams.lr,
                epsilon=hparams.adam_eps)
        train_op = optimizer.minimize(loss,global_step=global_step)

        return global_step,loss,train_op

class vgg13(Model):
    def __init__(self,hparams,num_classes):
        self.hparams = hparams
        self.num_classes=num_classes
        return
    
    def fprop(self,features):
        hparams = self.hparams
        x = tf.expand_dims(features,axis=3)
        
        x = tf.keras.layers.Conv2D(64, kernel_size=(3,3),padding='same',activation='relu',name='conv2d_1')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1,name='batch_normalization_1')(x)
        
        x = tf.keras.layers.Conv2D(64, kernel_size=(3,3),padding='same',activation='relu',name='conv2d_2')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1,name='batch_normalization_2')(x)
 
        x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
        
        x = tf.keras.layers.Conv2D(128, kernel_size=(3,3),padding='same',activation='relu',name='conv2d_3')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1,name='batch_normalization_3')(x)
        
        x = tf.keras.layers.Conv2D(128, kernel_size=(3,3),padding='same',activation='relu',name='conv2d_4')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1,name='batch_normalization_4')(x)
 
        x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
        
        x = tf.keras.layers.Conv2D(256, kernel_size=(3,3),padding='same',activation='relu',name='conv2d_5')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1,name='batch_normalization_5')(x)
        
        x = tf.keras.layers.Conv2D(256, kernel_size=(3,3),padding='same',activation='relu',name='conv2d_6')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1,name='batch_normalization_6')(x)
 
        x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)

        x = tf.keras.layers.Conv2D(512, kernel_size=(3,3),padding='same',activation='relu',name='conv2d_7')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1,name='batch_normalization_7')(x)
        
        x = tf.keras.layers.Conv2D(512, kernel_size=(3,3),padding='same',activation='relu',name='conv2d_8')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1,name='batch_normalization_8')(x)
 
        x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)

        x = tf.keras.layers.Conv2D(512, kernel_size=(3,3),padding='same',activation='relu',name='conv2d_9')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1,name='batch_normalization_9')(x)
        
        x = tf.keras.layers.Conv2D(512, kernel_size=(3,3),padding='same',activation='relu',name='conv2d_10')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1,name='batch_normalization_10')(x)
 
        x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)


        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(self.num_classes,name='dense_1')(x)
        return {self.O_LOGITS: x,
                self.O_PROBS: tf.nn.softmax(x)}




    def _conv_block(self,x,n_filters, kernel_size=(3,3),pool_size=(2,2), **kwargs):
        x = self._conv_bn(x, n_filters, kernel_size, **kwargs)
        x = self._conv_bn(x, n_filters, kernel_size, **kwargs)
        return tf.keras.layers.MaxPooling2D(pool_size=pool_size)(x)

    def _conv_bn(self,x, n_filters, kernel_size=(3,3), **kwargs):
        x = tf.keras.layers.Conv2D(n_filters,
                kernel_size=kernel_size,
                padding='same',
                activation='relu',
                **kwargs)(x)
        return tf.keras.layers.BatchNormalization(axis=-1)(x)






