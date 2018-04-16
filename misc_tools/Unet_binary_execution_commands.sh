prepare_data.py
nohup python train.py --device-ids 0,1 >train.log 2>&1 &
model_dir_name=unet_binary_<id>  # Change this while testing different models
TODO: Revisit output and input paths of generate_masks and evaluate
model_dir=data/models/$model_dir_name
mkdir $model_dir
cp runs/debug/model_0.pt $model_dir
prediction_dir=predictions/unet/$model_dir_name
mkdir -p $prediction_dir
python generate_masks.py --output_path $prediction_dir --model_type UNet --problem_type binary --model_path $model_dir --fold 0 --batch-size 4
python evaluate.py --target_path $prediction_dir --problem_type binary --train_path data/cropped_train
