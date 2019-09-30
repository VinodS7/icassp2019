num_models=3

if [ ! -d $models ]; then
      mkdir $models
  fi

for iteration in $(seq 1 $num_models) 
do
    mkdir models/model$iteration

done

echo "Script starts!"

for iteration in $(seq 1 $num_models)
do 
    echo "----------Training Model$iteration----------"
    nice python train.py --train_csv_file train.csv --train_audio_dir /import/c4dm-datasets/FSD/audio_train --save_model_dir models/model$iteration --label_data_file labels.npy
done
