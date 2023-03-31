#!/bin/bash

lr=$1 # 1e-4
epoch=$2 # 20
bs=$3  # 32

job=slurm_train
save_path=sp2023/hw7/work_dirs/rag_model_lr${lr}_epoch${epoch}_bs${bs}/
now=$(date +"%Y%m%d_%H%M%S")


mkdir -p $save_path

srun -p gpuA40x4 -n 1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=3 --job-name=$job --mem-per-cpu=16GB --time 00-20:00:00 -A bbrt-delta-gpu  \
python sp2023/hw7/classification_rag.py  \
--experiment "train" --graph-name ${save_path}/plot \
--device cuda \
--batch_size $bs --lr $lr --num_epochs $epoch 2>&1 | tee $save_path/$now.txt