#!/bin/bash
#SBATCH -p GPU # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -t 0-36:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --gres=gpu:1

eval "$(conda shell.bash hook)"
conda activate tuneocto
cd /home/u950323/octo
python octo_finetuning.py --pretrained_path=hf://rail-berkeley/octo-small-1.5 --data_dir='/home/u950323/octo/data_finetune' --save_dir='/home/u950323/trained-models/octo_checkpoints/' --batch_size=16 
