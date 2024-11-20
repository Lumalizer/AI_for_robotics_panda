#!/bin/bash
#SBATCH --job-name=octoFineTune
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00

eval "$(conda shell.bash hook)"
conda activate tuneocto
cd "$HOME/octo"

# Default values
PRETRAINED_PATH="hf://rail-berkeley/octo-small-1.5"
DATA_DIR="$HOME/tfds_datasets"
SAVE_DIR="$HOME/octo/trained_models/"
BATCH_SIZE=16
NAME=""

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --pretrained_path) PRETRAINED_PATH="$2"; shift ;;
    --data_dir) DATA_DIR="$2"; shift ;;
    --save_dir) SAVE_DIR="$2"; shift ;;
    --batch_size) BATCH_SIZE="$2"; shift ;;
    --name) NAME="$2"; shift ;;
    *) 
      echo "Invalid argument: $1"
      exit 1
  esac
  shift
done

# Ensure NAME is provided
if [ -z "$NAME" ]; then
  echo "Error: --name argument is required"
  exit 1
fi

# Combine NAME with DATA_DIR to make the new data directory
SAVE_DIR="$DATA_DIR/$NAME"

python finetuning_octo.py --pretrained_path=$PRETRAINED_PATH --data_dir=$DATA_DIR --save_dir=$SAVE_DIR --batch_size=$BATCH_SIZE