#!/bin/bash

model=$1 # distilbert-base-uncased
lr=$2 # 1e-4
epoch=$3 # 20
bs=$4  # 32

job=slurm_train
save_path=work_dirs/${model}_lr${lr}_epoch${epoch}_bs${bs}/
now=$(date +"%Y%m%d_%H%M%S")


mkdir -p $save_path

srun -p gpuA40x4 -n 1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=3 --job-name=$job --mem-per-cpu=16GB --time 00-20:00:00 -A bbrt-delta-gpu  \
python classification.py  \
--experiment "train" --graph-name ${save_path}/plot \
--device cuda --model $model \
--batch_size $bs --lr $lr --num_epochs $epoch 2>&1 | tee $save_path/$now.txt