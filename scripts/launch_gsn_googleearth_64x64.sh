#!/bin/bash

#SBATCH -o training_scripts/training_script.sh.log-%j
#SBATCH --gres=gpu:volta:1

# Loading the required module
# source /etc/profile
# source activate gsn
# module load anaconda/2022b
source activate /home/gridsan/fangd/.conda/envs/gsn

python3 --version

# Run the script
CUDA_VISIBLE_DEVICES=0 python train_gsn.py \
--base_config 'configs/models/gsn_googleearth_config.yaml' \
--log_dir 'logs' \
data_config.dataset='googleearth' \
data_config.data_dir='data/googleearth'
