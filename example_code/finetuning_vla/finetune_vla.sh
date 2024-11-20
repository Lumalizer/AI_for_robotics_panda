#!/bin/bash
#SBATCH --job-name=openVLAtuning
#SBATCH --gres=gpu:1          # Request 1 GPU
#SBATCH --mem=30G             # Request system memory (optional)
eval "$(conda shell.bash hook)"
conda activate openvla
torchrun --standalone --nnodes 1 --nproc-per-node 1 openvla/vla-scripts/finetune.py --name=diagnostic_one
