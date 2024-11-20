#!/bin/bash
#SBATCH --job-name=octoServer
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00

if [ -z "$1" ]; then
  echo "Usage: $0 <octo_path>"
  exit 1
fi

octo_path=$1

cd ~/octo
eval "$(conda shell.bash hook)"
conda activate tuneocto

port=8000

/usr/bin/ssh -i ~/ssh/ssh_key -N -f -R $port:localhost:$port portal.gpu4edu.uvt.nl
python server_octo.py $octo_path
