#!/bin/bash

#SBATCH -A cs601_gpu
#SBATCH --partition=mig_class
#SBATCH --reservation=MIG
#SBATCH --qos=qos_mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name="HW6 CS 601.471/671 homework"


module load anaconda

# init virtual environment if needed
# conda create -n toy_classification_env python=3.7

conda activate  pytorch # open the Python environment

# pip install -r requirements.txt # install Python dependencies

# runs your code
python classification.py  --experiment "overfit" --small_subset --device cuda --model "distilbert-base-uncased" --batch_size "32" --lr 1e-4 --num_epochs 20