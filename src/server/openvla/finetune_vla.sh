#!/bin/bash
#SBATCH --job-name=openVLAtuning
#SBATCH --gres=gpu:1          # Request 1 GPU
#SBATCH --mem=30G             # Request system memory (optional)

# Default arguments
JOB_NAME="openVLAtuning"
GRES="gpu:1"
MEM="30G"
SCRIPT_NAME="diagnostic_one"
DATA_ROOT_DIR="~/tfds_datasets"
RUN_ROOT_DIR="trained_models"
BATCH_SIZE=8
MAX_STEPS=200
SAVE_STEPS=20
IMAGE_AUG=true
SCRIPT_PATH="openvla/vla-scripts/finetune.py"

# Parse named arguments
while [ $# -gt 0 ]; do
  case "$1" in
    --job-name=*)
      JOB_NAME="${1#*=}"
      ;;
    --gres=*)
      GRES="${1#*=}"
      ;;
    --mem=*)
      MEM="${1#*=}"
      ;;
    --script-name=*)
      SCRIPT_NAME="${1#*=}"
      ;;
    --data_root_dir=*)
      DATA_ROOT_DIR="${1#*=}"
      ;;
    --run_root_dir=*)
      RUN_ROOT_DIR="${1#*=}"
      ;;
    --batch_size=*)
      BATCH_SIZE="${1#*=}"
      ;;
    --max_steps=*)
      MAX_STEPS="${1#*=}"
      ;;
    --save_steps=*)
      SAVE_STEPS="${1#*=}"
      ;;
    --image_aug=*)
      IMAGE_AUG="${1#*=}"
      ;;
    --script_path=*)
      SCRIPT_PATH="${1#*=}"
      ;;
    *)
      echo "Invalid argument: $1"
      exit 1
  esac
  shift
done

eval "$(conda shell.bash hook)"
conda activate openvla

# Use parsed or default arguments
torchrun --standalone --nnodes 1 --nproc-per-node 1 "$SCRIPT_PATH" --name="$SCRIPT_NAME" --data_root_dir="$DATA_ROOT_DIR" --run_root_dir="$RUN_ROOT_DIR" --batch_size="$BATCH_SIZE" --max_steps="$MAX_STEPS" --save_steps="$SAVE_STEPS" --image_aug="$IMAGE_AUG"
