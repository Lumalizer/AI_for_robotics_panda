#!/bin/bash
#SBATCH --job-name=openVLAserver
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00

# parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --openvla_path) openvla_path="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$openvla_path" ]; then
  echo "Usage: $0 --openvla_path <openvla_path>"
  exit 1
fi

cd ~/openvla
eval "$(conda shell.bash hook)"
conda activate openvla

port=8000

/usr/bin/ssh -i ~/ssh/ssh_key -N -f -R $port:localhost:$port portal.gpu4edu.uvt.nl
python server_openvla.py --openvla_path "$openvla_path"
