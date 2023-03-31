#!/bin/bash

job=slurm_train


srun -p gpuA40x4 -n 1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=3 --job-name=$job --mem-per-cpu=16GB --time 00-20:00:00 -A bbrt-delta-gpu  \
python sp2023/hw7/petals_infer.py