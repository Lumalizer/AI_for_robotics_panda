#!/bin/bash
#SBATCH -p GPU # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -t 0-36:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --gres=gpu:1

eval "$(conda shell.bash hook)"
conda activate tuneocto
cd ~
cd octo

# Default values
PRETRAINED_PATH="hf://rail-berkeley/octo-small-1.5"
DATA_DIR="~/tfds_datasets"
SAVE_DIR="trained_models"
BATCH_SIZE=16

# Parse named arguments
while [ $# -gt 0 ]; do
  case "$1" in
    --pretrained_path=*)
      PRETRAINED_PATH="${1#*=}"
      ;;
    --data_dir=*)
      DATA_DIR="${1#*=}"
      ;;
    --save_dir=*)
      SAVE_DIR="${1#*=}"
      ;;
    --batch_size=*)
      BATCH_SIZE="${1#*=}"
      ;;
    *)
      echo "Invalid argument: $1"
      exit 1
  esac
  shift
done

python octo_finetuning.py --pretrained_path=$PRETRAINED_PATH --data_dir=$DATA_DIR --save_dir=$SAVE_DIR --batch_size=$BATCH_SIZE
