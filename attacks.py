import tensorflow as tf
import numpy as np

import carlini_wagner_attack as CW



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


    graph_substitute = tf.Graph()
    graph_target = tf.Graph()

    with graph_substitute.as_default() and tf.Session(graph=graph_substitute) as sess:
        substitute = model.BaselineCNN(hparams,num_classes)
        cw = CW.CarliniWagnerAttack(model=substitute_model,save_model_dir=flags.save_model_dir,sess=sess,hparams=hparams)
        cw.build_attack()

        for i in range(10):
            data,_ = utils_tf._preprocess_data(flags.infer_audio_dir,file_names[i])

            audio,preds,snr = cw.attack(data)
      

if __name__=="__main__":
    main()
