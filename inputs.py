from __future__ import absolute_import,print_function,division

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
"""
1. Extract features from audio files that are given by an input csv file. 
2. Create labels for training using the target model 
3. Create iterator to shuffle and batch dataset for training
"""

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_path',None,'model path for the target model')
flags.DEFINE_string('csv_file',None,'csv file for the audio files')
flags.DEFINE_string('audio_path',None,'Path to the FSDKaggle2018 dataset')

def feature_extraction(clip,audio_path,hparams):
    """
    Extract features from a wavfile
    """
    clip_path = tf.string_join([audio_path,clip], separator=os.sep)
    clip_data = tf.read_file(clip_path)
    waveform,sr = tf.contrib.framework.python.ops.devode_wav(clip_data)
    check_sr = tf.assert_equal(sr, SAMPLE_RATE)

    check_channels = tf.assert_equal(tf.shape(waveform)[1],1)
    with tf.control_dependencies([tf.group(check_sr, check_channels)]):
        waveform = tf.squeeze(waveform)
    

    window_size = hparams.window_size
    hop_size = hparams.hop_size
    fft_length = hparams.fft_length
    magnitude_spectrogram = tf.abs(tf.contrib.signal.stft(signals=waveform,
                            frame_length=window_size,
                            frame_step=hop_size,
                            fft_length=fft_length))

    num_bins = fft_length//2 + 1
    mel_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
                            num_mel_bins=hparams.mel_bands,
                            num_spectrogram_bins=num_bins,
                            sample_rate=SAMPLE_RATE,
                            lower_edge_hertz=hparams.mel_min_hz,
                            upper_edge_hertz=hparams.mel_max_hz)
    mel_spectrogram = tf.matmul(magnitude_spectrogram,mel_matrix)
    log_mel_spectrogram = tf.log(mel_spectrogram + hparams.mel_log_offset)
    
    spectrogram_sr = 1 /hparams.stft_hop_seconds
    example_window_length_samples = (int(round(spectrogram_sr*hparams.example_window_seconds)))
    example_hop_length_samples = int(round(spectrogram_sr*hparams.example_hop_seconds))

    features = tf.contrib.signal.frame(
            signal=log_mel_spectrogram,
            frame_length=example_window_length_samples,
            frame_step=example_hop_length_samples,
            axis=0)
    return features

def label_data(model_path,csv_file,audio_path):
    """
    Label the data using a particular model and save the softmax values.
    Generates one softmax values per file
    """
    
    sr=32000
    df = pd.read_csv(csv_file)
    x,_ = utils_tf._load_dataset(cfg.to_dataset('training'))
    generator = utils.fit_scaler(x)
    file_names = df.iloc[:,0].values
    print(file_names)
    with tf.Graph().as_default() as graph:
        mel_filt = librosa.filters.mel(sr=32000,n_fft=1024,n_mels=64).T
        mel_filt = tf.convert_to_tensor(mel_filt,dtype=tf.float32)
        pcm = tf.placeholder(tf.float32,shape=[None],name='input_audio')
        model = CleverHansModel(model_path + '.meta',sr,generator,mel_filt)
        saver = model.build_graph(pcm)

    probs = []
    temp = {}
    with tf.Session(graph=graph) as sess:
        saver.restore(sess,model_path)
        print(len(file_names)) 
        for i in range(len(file_names)):
            data,_ = utils_tf._preprocess_data(audio_path,file_names[i])
            l = sess.run([model.get_probs()],feed_dict={pcm:data})
            l = np.squeeze(l)
            if(l.ndim !=1):
                l = np.mean(l,axis=0)

            temp[file_names[i]] = l
            print(i)
        print(temp)

    file = open('label_data','wb')

    pickle.dump(temp,file)
    file.close()


            
    return
def get_data(csv_record,audio_path,hparams):
    [clip,_] = tf.decode_csv(csv_record,record_defaults=[[''],[''],[0]])

    features = feature_extraction(clip,audio_path,hparams)

    label = data[clip] 
    num_examples = tf.shape(features)[0]
    labels = tf.tile([labels],[num_examples, 1])
    
    return

def dataset_iterator(csv_file,audio_path,hparams):
    """
    Create an iterator for the training process
    """
    dataset = tf.data.TextLineDataset(csv_file)

    dataset.skip(1)

    dataset.shuffle(buffer_size=10000)

    dataset = dataset.map(
            map_func=functools.partial(get_data,
                    clip_dir=audio_path,
                    hparams=hparams,
                    num_classes=num_classes),
            num_parallel_calls=2)
    dataset.apply(tf.contrib.data.unbatch())
    dataset = dataset.shuffle(buffer_size=20000)
    dataset = dataset.repeat(5)
    dataset.apply(tf.contrib.data.batch_drop_remainder(batch_size=hparams.batch_size))

    dataset = dataset.prefetch(10)

    iterator = dataset.make_initializable_iterator()
    features,lables = iterator.get_next()

    return features, labels, num_classes, iterator.initializer

    
    return


if __name__=="__main__":
    label_data(FLAGS.model_path,FLAGS.csv_file,FLAGS.audio_path)