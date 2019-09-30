import tensorflow as tf
import numpy as np
import argparse

import model
import inputs
"""
Training the substitute  model using the target  models soft labels
"""

def parse_flags():
    parser = argparse.ArgumentParser(description='Parser for training')
    parser.add_argument('--train_csv_file',type=str,default='',
            help='Path to csv file containing names of files')
    parser.add_argument('--train_audio_dir',type=str,default='',
            help='Path to directory containing audio files')
    parser.add_argument('--save_model_path',type=str,default='',
            help='Path to directory to save model variables')
    parser.add_argument('--label_data_file',type=str,default='',
            help='Path to file containing data labeled by target model')
    parser.add_argument('--hparams',type=str,default='',
            help='model hyperparameters in comma-separated name=values format')
    flags = parser.parse_args()
    
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
            batch_size=64,
            weights_init_stddev=1e-3,
            lr=1e-4,
            adam_eps=1e-8)
    hparams.parse(flag_hparams)
    return hparams
def main():
    flags = parse_flags()
    hparams= parse_hparams(flags.hparams)
    num_classes=41
    print(hparams)
    with tf.Graph().as_default():
        substitute = model.BaselineCNN(hparams,num_classes)
        features,labels,num_classes,input_init=inputs.dataset_iterator(flags.train_csv_file,flags.train_audio_dir,flags.label_data_file,hparams)
        global_step,loss,train_op = substitute.train(features,labels)
        saver = tf.train.Saver()
        saver_hook = tf.train.CheckpointSaverHook(save_steps=250,checkpoint_dir=flags.save_model_path,saver=saver)
        summary_op = tf.summary.merge_all()
        summary_hook = tf.train.SummarySaverHook(save_steps=500,output_dir=flags.save_model_path,summary_op=summary_op)




        with tf.train.SingularMonitoredSession(hooks=[saver_hook,summary_hook],
                                            checkpoint_dir=flags.save_model_path) as sess:
            sess.raw_session().run(input_init)

            while not sess.should_stop():
                step,_,l = sess.run([global_step,train_op,loss])
                print(step,l)
             
if __name__ == '__main__':
    main()
