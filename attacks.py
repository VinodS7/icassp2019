import os
import subprocess
import time

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
            max_iterations=5000,
            const=1e-2,
            sample_rate=32000,
            targeted=True,
            vgg13_features=False)

    hparams.parse(flag_hparams)
    return hparams



def main():
    flags = parse_flags()
    hparams = parse_hparams(flags.hparams)
    print(hparams)
    num_classes=41
    df = pd.read_csv(flags.infer_csv_file)
    file_names = df.iloc[:,0].values
    labels = df.iloc[:,1].values
    
    gt = []
    orig_label = []
    adv_label = []
    orig_pred = []
    adv_pred = []
    snr = []
    files = []
    t = []
    with tf.Graph().as_default() and tf.Session() as sess:
        if(hparams.vgg13_features):
            substitute_model = model.vgg13(hparams,num_classes)
        else:
            substitute_model = model.BaselineCNN(hparams,num_classes)
        cw = CW.CarliniWagnerAttack(model=substitute_model,save_model_dir=flags.save_model_dir,sess=sess,hparams=hparams)
        cw.build_attack()

        for i in range(len(file_names)):
            start_time=time.time()
            #call = ['ffmpeg','-v','quiet','-i',os.path.join(flags.infer_audio_dir,file_names[i]),'-f','f32le', '-ar',str(hparams.sample_rate),'-ac','1','pipe:1']
            #samples = subprocess.check_output(call)
            #data = np.frombuffer(samples, dtype=np.float32)
            data,_ = utils_tf._preprocess_data(flags.infer_audio_dir,file_names[i])                
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
            
            audio,pred,pred_orig,noise = cw.attack(data,target_label)
            
            if(audio is None):
                continue
            print('TIME IN SECONDS!',time.time()-start_time)
            gt.append(lab)
            orig_label.append(np.argmax(pred_orig))
            orig_pred.append(np.max(pred_orig))
            adv_label.append(np.argmax(pred))
            adv_pred.append(np.max(pred))
            snr.append(noise)
            files.append(file_names[i])
            t.append(time.time()-start_time)
            wav.write(os.path.join(flags.write_audio_dir,file_names[i]),44100,audio) 
        
        df_out = pd.DataFrame({'fname':files,'gt':gt,'original_label':orig_label,'original_pred':
            orig_pred,'adv_label':adv_label,'adv_pred':adv_pred,'snr':snr,'time':t})
        df_out.to_csv('adv_data_vgg13.csv',index=False)

if __name__=="__main__":
    main()
