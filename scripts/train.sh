#!/bin/bash

#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --job-name=train
#SBATCH --cpus-per-task=10
#SBATCH --time=60:00:00
#SBATCH -C TitanRTX
#SBATCH --output=train_separateProcessing_%A.out

# Activate environment
module load cuda12.3/toolkit
source activate /var/scratch/avasilco/conda/envs/latte

export HF_HOME=/var/scratch/avasilco/hf_home
export TRANSFORMERS_CACHE=/var/scratch/avasilco/hf_home/transformers
export HUGGINGFACE_HUB_CACHE=/var/scratch/avasilco/hf_home/hub
export HF_DATASETS_CACHE=/var/scratch/avasilco/hf_home/datasets
export XDG_CACHE_HOME=/var/scratch/avasilco/hf_home/xdg
export HOME=/var/scratch/avasilco

# Turn on unbuffered output so you see prints immediately.
export PYTHONUNBUFFERED=1

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export NODE_RANK=$SLURM_NODEID
export NPROC_PER_NODE=1
export NNODES=1

# Run code
torchrun \
  --nproc_per_node $NPROC_PER_NODE \
  --nnodes $NNODES \
  --node_rank $NODE_RANK \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
  --rdzv_backend c10d \
  train.py \
    --latent_dir_train "../GenImage/GenImage/stable_diffusion_v_1_4/latents_train" \
    --latent_dir_validation "../GenImage/GenImage/stable_diffusion_v_1_4/latents_val" \
    --model_traj "TemporalCLIPLatentClassifier" \
    --clip_type "convnext_base_in22k" \
    --tracked_timesteps "[981, 741, 521, 261, 1]" \
    --isTrain 1 \
    --num_class 2 \
    --data_size 224 224 \
    --batch_size 32 \
    --exp_name "SDV14_convNext_separateProcessing" \
    --epochs 12 \
    --is_amp \
    --is_warmup \
    --qkNorm