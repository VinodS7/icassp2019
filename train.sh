num_models=3

if [ ! -d vgg13_features_models ]; then
      mkdir vgg13_features_models
  fi

for iteration in $(seq 1 $num_models) 
do
    mkdir vgg13_features_models/model$iteration

done

echo "Script starts!"

for iteration in $(seq 1 $num_models)
do 
    echo "----------Training Model$iteration----------"
    python train.py --train_csv_file train.csv --train_audio_dir /import/c4dm-datasets/FSD/audio_train --save_model_dir vgg13_features_models/model$iteration --label_data_file labels.npy --hparams sample_rate=32000
done
