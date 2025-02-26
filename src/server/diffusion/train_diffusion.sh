#!/bin/bash
#SBATCH --job-name=diffusion
#SBATCH --gres=gpu:1          # Request 1 GPU
#SBATCH --mem=40G             # Request system memory (optional)

# Default arguments
NAME="diffusion"
DATA_ROOT_DIR="$HOME/lerobot_datasets"
RUN_ROOT_DIR="$HOME/diffusion/trained_models"
SCRIPT_PATH="$HOME/diffusion/train_diffusion.py"
DATASET_FPS=15
DELTAS_LIMIT=7
BATCH_SIZE=32
MAX_STEPS=5000

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --name) NAME="$2"; shift ;;
    --data_root_dir) DATA_ROOT_DIR="$2"; shift ;;
    --run_root_dir) RUN_ROOT_DIR="$2"; shift ;;
    --batch_size) BATCH_SIZE="$2"; shift ;;
    --max_steps) MAX_STEPS="$2"; shift ;;
    --dataset_fps) DATASET_FPS="$2"; shift ;;
    --deltas_limit) DELTAS_LIMIT="$2"; shift ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --name: Name of the model (default: $NAME)"
      echo "  --data_root_dir: Path to the data directory (default: $DATA_ROOT_DIR)"
      echo "  --run_root_dir: Path to the run directory (default: $RUN_ROOT_DIR)"
      echo "  --batch_size: Batch size (default: $BATCH_SIZE)"
      echo "  --max_steps: Maximum steps (default: $MAX_STEPS)"
      echo "  --dataset_fps: Dataset FPS (default: $DATASET_FPS)"
      echo "  --deltas_limit: Deltas limit (default: $DELTAS_LIMIT)"
      exit 0
      ;;
    *)
      echo "Invalid argument: $1"
      exit 1
  esac
  shift
done

# Combine NAME with RUN_ROOT_DIR and other variables to make the new save directory
LONGNAME="$NAME-batch_$BATCH_SIZE-dataset_fps_$DATASET_FPS-deltas_limit_$DELTAS_LIMIT"
RUN_ROOT_DIR="$RUN_ROOT_DIR/$LONGNAME"
DATA_ROOT_DIR="$DATA_ROOT_DIR/$NAME"

eval "$(conda shell.bash hook)"
conda activate diffusion

# Use parsed or default arguments
python "$SCRIPT_PATH" --name="$LONGNAME" --data_root_dir="$DATA_ROOT_DIR" --run_root_dir="$RUN_ROOT_DIR" --batch_size="$BATCH_SIZE" --max_steps="$MAX_STEPS" --dataset_fps="$DATASET_FPS" --deltas_limit="$DELTAS_LIMIT"