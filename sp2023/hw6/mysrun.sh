#!/bin/bash

job=slurm_train
save_path=work_dirs/
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p $save_path

srun  -p $2 -N 1 -n $1 --gres=gpu:$1 --ntasks-per-node=$1 --job-name=$job --mem-per-cpu=10GB --time 00-10:00:00 -A hwang9_gpu \
python classification.py  \
--experiment "overfit" --small_subset \
--device cuda --model "distilbert-base-uncased" \
--batch_size "32" --lr 1e-4 --num_epochs 20 2>&1 | tee $save_path/$now.txt