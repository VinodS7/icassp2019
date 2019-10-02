import tensorflow as tf
import numpy as np
import pandas as pd
import argparse

import model
import inputs
import utils_tf

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
    flags=parser.parse_args()
    return flags

def gen_features(file_name,train_audio_dir,hparams):
    """
    function to generate predictions for individual audio files
    """

    features = inputs.feature_extraction(file_name,train_audio_dir,hparams)
    
    return features

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
        file_name = tf.placeholder(tf.string)
        start_vars = set(x.name for x in tf.global_variables())
        features = gen_features(file_name,flags.infer_audio_dir,hparams)
        preds = substitute.get_probs(features) 
        end_vars = tf.global_variables()
        model_vars = [x for x in end_vars if x.name not in start_vars]
        saver = tf.train.Saver(var_list=model_vars)
        
        with tf.Session() as sess:
            saver.restore(sess=sess,save_path=flags.save_model_dir)
             
            for i in range(len(file_names)):
                pr = sess.run([preds],feed_dict={file_name:file_names[i]})
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
if __name__=='__main__':
    main()
