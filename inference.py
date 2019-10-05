import os
import subprocess

import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
import librosa

from cleverhans_wrapper import CleverHansModel
import model
import inputs
import utils_tf
import utils
import config as cfg

def parse_flags():
    parser = argparse.ArgumentParser(description='Parser for inference and evaluation')
    parser.add_argument('--infer_csv_file',type=str,default='',
            help='Path to csv file containing names of files')
    parser.add_argument('--infer_audio_dir',type=str,default='',
            help='Path to audio directory containing audio files')
    parser.add_argument('--save_model_dir',type=str,default='',
            help='Path to directory that contains the saved model params')
    parser.add_argument('--hparams',type=str,default='',
            help='model hyperparameters in comma-separated name=values format')
    parser.add_argument('--target_model',type=bool,default=False,
            help='target the model')
    flags=parser.parse_args()
    return flags


def parse_hparams(flag_hparams):
    hparams = tf.contrib.training.HParams(
            stft_window_seconds=0.025,
            stft_hop_seconds=0.010,
            mel_bands=64,
            mel_min_hz=125,
            mel_max_hz=7500,
            mel_log_offset=0.001,
            example_window_seconds=0.250,
            example_hop_seconds=0.125,
            weights_init_stddev=1e-3)

    hparams.parse(flag_hparams)
    return hparams

def main():

    flags = parse_flags()
    hparams = parse_hparams(flags.hparams)
    num_classes=41
    df = pd.read_csv(flags.infer_csv_file)
    file_names = df.iloc[:,0].values
    count=0
    
    with tf.Graph().as_default():
        substitute = model.BaselineCNN(hparams,num_classes)
        audio_input = tf.placeholder(tf.float32,shape=[None],name='audio_input')
        start_vars = set(x.name for x in tf.global_variables())
        features = inputs.compute_features(audio_input,hparams)
        preds = substitute.get_probs(features) 
        end_vars = tf.global_variables()
        model_vars = [x for x in end_vars if x.name not in start_vars]
        saver = tf.train.Saver(var_list=model_vars)
        
        with tf.Session() as sess:
            saver.restore(sess=sess,save_path=flags.save_model_dir)
             
            for i in range(100):
                call = ['ffmpeg','-v','quiet','-i',os.path.join(flags.infer_audio_dir,file_names[i]),'-f','f32le','-ar',str(44100),'-ac','1','pipe:1']
                samples = subprocess.check_output(call)
                waveform = np.frombuffer(samples, dtype=np.float32)
    
                pr = sess.run([preds],feed_dict={audio_input:waveform})
                pr = np.squeeze(pr)
                if(pr.ndim != 1):
                    pr = np.mean(pr,axis=0)
                #print(pr)
                #print(np.argmax(pr),np.max(pr))
                #print(df.iloc[i,1],utils_tf._convert_label_to_label_name(int(np.argmax(pr))))
                lab = utils_tf._convert_label_name_to_label(df.iloc[i,1])
                if(lab==np.argmax(pr)):
                    count+=1
                    print(lab,np.argmax(pr),np.max(pr))
            print(float(count/len(file_names)))
    return

def target():
    """
    Label the data using a particular model and save the softmax values.
    Generates one softmax values per file
    """
    flags = parse_flags()
    hparams = parse_hparams(flags.hparams)
    num_classes=41
    df = pd.read_csv(flags.infer_csv_file)
    file_names = df.iloc[:,0].values
    
    count=0
     
    sr=32000
    df = pd.read_csv(flags.infer_csv_file)
    x,_ = utils_tf._load_dataset(cfg.to_dataset('training'))
    generator = utils.fit_scaler(x)
    file_names = df.iloc[:,0].values
    with tf.Graph().as_default() as graph:
        mel_filt = librosa.filters.mel(sr=32000,n_fft=1024,n_mels=64).T
        mel_filt = tf.convert_to_tensor(mel_filt,dtype=tf.float32)
        pcm = tf.placeholder(tf.float32,shape=[None],name='input_audio')
        model = CleverHansModel(flags.save_model_dir + '.meta',sr,generator,mel_filt)
        saver = model.build_graph(pcm)

    with tf.Session(graph=graph) as sess:
        saver.restore(sess,flags.save_model_dir)
        print(len(file_names)) 
        for i in range(100):
            data,_ = utils_tf._preprocess_data(flags.infer_audio_dir,file_names[i])
            l = sess.run([model.get_probs()],feed_dict={pcm:data})
            l = np.squeeze(l)
            if(l.ndim !=1):
                l = np.mean(l,axis=0)
            
            lab = utils_tf._convert_label_name_to_label(df.iloc[i,1])
            if(lab==np.argmax(l)):
                count+=1
                print(lab,np.argmax(l))

            print(count/100)
                
if __name__=='__main__':
        target()
