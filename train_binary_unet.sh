#!/bin/sh
set -e
primary_train_result_dir='runs'
usecase='binary'
model='UNet'
run_tag=''
job_dir=$(python -c "import utils; print(utils.get_run_dir_from_args('$primary_train_result_dir','$usecase','$model','$run_tag'))")
echo "writing results to job dir: $job_dir"
for i in 0 1 2 3
do
   echo "starting training for fold $i"
   task_dir="$job_dir/fold_$i"
   echo "writing results to task dir: $task_dir"
   python train.py --fold $i --lr 0.0001 --n-epochs 10 --type $usecase --jaccard-weight 1 --model $model --reuse-train-dir "$task_dir"
   python train.py --fold $i --lr 0.00001 --n-epochs 20 --type $usecase --jaccard-weight 1 --model $model --reuse-train-dir "$task_dir"
done
