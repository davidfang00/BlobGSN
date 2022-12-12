#!/bin/bash

#SBATCH -o walkthrough_scripts/walkthrough_scripts.sh.log-%j
#SBATCH --gres=gpu:volta:1

# Loading the required module
# source /etc/profile
# source activate gsn
# module load anaconda/2022b
source activate /home/gridsan/fangd/.conda/envs/gsn

python3 --version

# Run the script
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python walkthrough_script.py

