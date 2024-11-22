#!/bin/bash
#SBATCH --job-name=octoFineTune
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00

eval "$(conda shell.bash hook)"
conda activate tuneocto
cd "$HOME/octo"

# Default values
MODEL_SIZE="small"
DATA_DIR="$HOME/tfds_datasets"
SAVE_DIR="$HOME/octo/trained_models/"
BATCH_SIZE=16
NAME=""
ACTION_HORIZON=4
WINDOW_SIZE=2
STEPS=20000

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --model_size) MODEL_SIZE="$2"; shift ;;
    --data_dir) DATA_DIR="$2"; shift ;;
    --save_dir) SAVE_DIR="$2"; shift ;;
    --batch_size) BATCH_SIZE=$2; shift ;;
    --name) NAME="$2"; shift ;;
    --action_horizon) ACTION_HORIZON=$2; shift ;;
    --window_size) WINDOW_SIZE=$2; shift ;;
    --steps) STEPS=$2; shift ;;
    --help) 
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --name: Name of the model (required)"
      echo "  --model_size: Size of the model [small / base] (default: $MODEL_SIZE)"
      echo "  --data_dir: Path to the data directory (default: $DATA_DIR)"
      echo "  --save_dir: Path to the save directory (default: $SAVE_DIR)"
      echo "  --batch_size: Batch size (default: $BATCH_SIZE)"
      echo "  --action_horizon: Action horizon (default: $ACTION_HORIZON)"
      echo "  --window_size: Window size (default: $WINDOW_SIZE)"
      echo "  --steps: Number of steps (default: $STEPS)"
      exit 0
      ;;
    *) 
      echo "Invalid argument: $1"
      exit 1
  esac
  shift
done


if [ "$MODEL_SIZE" == "small" ]; then
  PRETRAINED_PATH="hf://rail-berkeley/octo-small-1.5"
elif [ "$MODEL_SIZE" == "base" ]; then
  PRETRAINED_PATH="hf://rail-berkeley/octo-base-1.5"
else
  echo "Invalid model size: $MODEL_SIZE"
  exit 1
fi

# Ensure NAME is provided
if [ -z "$NAME" ]; then
  echo "Error: --name argument is required"
  exit 1
fi

# Combine NAME with SAVE_DIR and other variables eg window size etc to make the new save directory
LONGNAME="$NAME-model_$MODEL_SIZE-horizon_$ACTION_HORIZON-window_$WINDOW_SIZE-batch_$BATCH_SIZE"
SAVE_DIR="$SAVE_DIR/$LONGNAME"

python finetuning_octo.py --pretrained_path=$PRETRAINED_PATH --data_dir=$DATA_DIR --save_dir=$SAVE_DIR --batch_size=$BATCH_SIZE --action_horizon=$ACTION_HORIZON --window_size=$WINDOW_SIZE --steps=$STEPS --wandb_name=$LONGNAME