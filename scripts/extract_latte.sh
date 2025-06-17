#!/bin/bash

#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --job-name=latents
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
#SBATCH --output=extract_latents_%A.out

export HF_HOME=/var/scratch/avasilco/hf_home
export TRANSFORMERS_CACHE=/var/scratch/avasilco/hf_home/transformers
export HUGGINGFACE_HUB_CACHE=/var/scratch/avasilco/hf_home/hub
export HF_DATASETS_CACHE=/var/scratch/avasilco/hf_home/datasets
export XDG_CACHE_HOME=/var/scratch/avasilco/hf_home/xdg
export HOME=/var/scratch/avasilco

# unbuffered Python output
export PYTHONUNBUFFERED=1

# Activate environment
module load cuda12.3/toolkit
source activate /var/scratch/avasilco/conda/envs/latte

python extract_latte.py \
    --real_folders "../GenImage/GenImage/stable_diffusion_v_1_4/val/nature" \
    --fake_folders "../GenImage/GenImage/stable_diffusion_v_1_4/val/ai" \
    --cache_dirs   "../GenImage/GenImage/stable_diffusion_v_1_4/latents_val" \
    --data_size    224 224 \