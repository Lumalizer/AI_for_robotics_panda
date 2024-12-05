#!/bin/bash
#SBATCH --job-name=octoServer
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --octo_path) octo_path="$2"; shift ;;
        --window_size) window_size="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$octo_path" ]; then
  echo "Usage: $0 --octo_path <octo_path> --window_size <window_size>"
  exit 1
fi

if [ -z "$window_size" ]; then
  echo "Usage: $0 --octo_path <octo_path> --window_size <window_size>"
  exit 1
fi

cd ~/octo
eval "$(conda shell.bash hook)"
conda activate tuneocto

port=8000

/usr/bin/ssh -i ~/ssh/ssh_key -N -f -R $port:localhost:$port portal.gpu4edu.uvt.nl
python server_octo.py --octo_path $octo_path --window_size $window_size