import os
import subprocess

import tensorflow as tf
import numpy as np
import argparse
import pandas as pd
import scipy.io.wavfile as wav

import model
import carlini_wagner_attack as CW
import utils_tf
import inputs

def parse_flags():
    parser = argparse.ArgumentParser(description='Parser for inference and evaluation')
    parser.add_argument('--infer_csv_file',type=str,default='',
            help='Path to csv file containing names of files')
    parser.add_argument('--write_audio_dir',type=str,default='',
            help='Path to write adversarial audio files')
    parser.add_argument('--infer_audio_dir',type=str,default='',
            help='Path to audio directory containing audio files')
    parser.add_argument('--save_model_dir',type=str,default='',
            help='Path to directory that contains the saved model params')
    parser.add_argument('--hparams',type=str,default='',
            help='model hyperparameters in comma-separated name=values format')
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
            weights_init_stddev=1e-3,
            learning_rate=5e-4,
            confidence=0,
            const=1e-2,
            targeted=True)

    hparams.parse(flag_hparams)
    return hparams



def main():
    flags = parse_flags()
    hparams = parse_hparams(flags.hparams)
    num_classes=41
    df = pd.read_csv(flags.infer_csv_file)
    file_names = df.iloc[:,0].values
    labels = df.iloc[:,1].values
    print(hparams)
    with tf.Graph().as_default() and tf.Session() as sess:
        substitute_model = model.BaselineCNN(hparams,num_classes)
        cw = CW.CarliniWagnerAttack(model=substitute_model,save_model_dir=flags.save_model_dir,sess=sess,hparams=hparams)
        cw.build_attack()

        for i in range(20):
            call = ['ffmpeg','-v','quiet','-i',os.path.join(flags.infer_audio_dir,file_names[i]),'-f','f32le',
                    '-ar',str(44100),'-ac','1','pipe:1']
            samples = subprocess.check_output(call)
            data = np.frombuffer(samples, dtype=np.float32)
                            
            lab = utils_tf._convert_label_name_to_label(labels[i])
            print(data.shape)
            
            if(hparams.targeted):
                set_target=False

                while(not set_target):
                    target_label = np.random.randint(41)
                    if(lab != target_label):
                        set_target=True
            else:
                target_label = lab
            

            audio,preds,snr = cw.attack(data,target_label)
            print(audio.shape,data.shape)
            wav.write(os.path.join(flags.write_audio_dir,file_names[i]),44100,audio) 
            wav.write(os.path.join(flags.write_audio_dir,'orig-'+file_names[i]),44100,data) 


if __name__=="__main__":
    main()
