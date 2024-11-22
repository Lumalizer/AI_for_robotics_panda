#!/bin/bash
#SBATCH --job-name=openVLAtuning
#SBATCH --gres=gpu:1          # Request 1 GPU
#SBATCH --mem=40G             # Request system memory (optional)

# Default arguments
NAME="diagnostic_one"
DATA_ROOT_DIR="$HOME/tfds_datasets"
RUN_ROOT_DIR="$HOME/openvla/trained_models"
SCRIPT_PATH="$HOME/openvla/finetune_vla.py"
BATCH_SIZE=8
MAX_STEPS=200
SAVE_STEPS=40
IMAGE_AUG="true"

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --name) NAME="$2"; shift ;;
    --data_root_dir) DATA_ROOT_DIR="$2"; shift ;;
    --run_root_dir) RUN_ROOT_DIR="$2"; shift ;;
    --batch_size) BATCH_SIZE="$2"; shift ;;
    --max_steps) MAX_STEPS="$2"; shift ;;
    --save_steps) SAVE_STEPS="$2"; shift ;;
    --image_aug) IMAGE_AUG="$2"; shift ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --name: Name of the model (default: $NAME)"
      echo "  --data_root_dir: Path to the data directory (default: $DATA_ROOT_DIR)"
      echo "  --run_root_dir: Path to the run directory (default: $RUN_ROOT_DIR)"
      echo "  --batch_size: Batch size (default: $BATCH_SIZE)"
      echo "  --max_steps: Maximum steps (default: $MAX_STEPS)"
      echo "  --save_steps: Save steps (default: $SAVE_STEPS)"
      echo "  --image_aug: Image augmentation (default: $IMAGE_AUG)"
      exit 0
      ;;
    *)
      echo "Invalid argument: $1"
      exit 1
  esac
  shift
done

# Combine NAME with RUN_ROOT_DIR and other variables to make the new save directory
LONGNAME="$NAME-batch_$BATCH_SIZE-image_aug_$IMAGE_AUG"
RUN_ROOT_DIR="$RUN_ROOT_DIR/$LONGNAME"

eval "$(conda shell.bash hook)"
conda activate openvla

# Use parsed or default arguments
torchrun --standalone --nnodes 1 --nproc-per-node 1 "$SCRIPT_PATH" --data_root_dir="$DATA_ROOT_DIR" --run_root_dir="$RUN_ROOT_DIR" --batch_size="$BATCH_SIZE" --max_steps="$MAX_STEPS" --save_steps="$SAVE_STEPS" --image_aug="$IMAGE_AUG"
