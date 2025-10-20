#!/bin/bash

#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=test
#SBATCH --cpus-per-task=9
#SBATCH --time=10:00:00
#SBATCH --output=test_separateProcessing_%A.out

# Activate environment
module load cuda12.3/toolkit
source activate /var/scratch/avasilco/conda/envs/latte

export HF_HOME=/var/scratch/avasilco/hf_home
export TRANSFORMERS_CACHE=/var/scratch/avasilco/hf_home/transformers
export HUGGINGFACE_HUB_CACHE=/var/scratch/avasilco/hf_home/hub
export HF_DATASETS_CACHE=/var/scratch/avasilco/hf_home/datasets
export XDG_CACHE_HOME=/var/scratch/avasilco/hf_home/xdg
export HOME=/var/scratch/avasilco

# Run code
srun python test_models.py \
    --latent_dirs_test "../GenImage/GenImage/ADM/latents_val" "../GenImage/GenImage/BigGAN/latents_val" "../GenImage/GenImage/glide/latents_val" "../GenImage/GenImage/Midjourney/latents_val" "../GenImage/GenImage/stable_diffusion_v_1_4/latents_val" "../GenImage/GenImage/stable_diffusion_v_1_5/latents_val" "../GenImage/GenImage/VQDM/latents_val" "../GenImage/GenImage/wukong/latents_val" \
    --method_names "ADM" "BigGAN" "glide" "Midjourney" "SDV1.4" "SDV1.5" "VQDM" "wukong" \
    --checkpoint "checkpoints/SDV14_convNext_separateProcessing/ACC_best.pth" \
    --model_type "LatentTrajectoryClassifier" \
    --clip_type "convnext_base_in22k" \
    --tracked_timesteps "[981, 741, 521, 261, 1]" \
    --exp_name "SDV14_convNext_separateProcessing" \