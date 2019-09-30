from __future__ import absolute_import, print_function,division

import os

import librosa
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

import utils_tf
import utils
import config as cfg
from cleverhans_wrapper import CleverHansModel
import inputs


csv_file = 'train.csv'
audio_path = '/import/c4dm-datasets/FSD/audio_train/'

hparams = tf.contrib.training.HParams(
        stft_window_seconds=0.025,
        stft_hop_seconds=0.010,
        mel_bands=64,
        mel_min_hz=125,
        mel_max_hz=7500,
        mel_log_offset=0.001,
        example_window_seconds=0.250,
        example_hop_seconds=0.125,
        batch_size=64,
        weights_init_stddev=1e-3,
        lr=1e-4,
        adam_spe=1e-8,
        classifier='softmax')
graph = tf.Graph()

with graph.as_default():
    features,label,num_classes,input_init = inputs.dataset_iterator(csv_file,audio_path,hparams)

with tf.Session(graph=graph) as sess:
    sess.run([input_init])
    
    tf.tables_initializer().run()
    for i in range(10):
        f,l = sess.run([features,label])
        f = np.squeeze(f)
        l - np.squeeze(l) print(f.shape,l.shape)
        print(l[:,0])
