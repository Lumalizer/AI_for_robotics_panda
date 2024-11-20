#!/bin/bash
#SBATCH --job-name=openVLAserver
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00

cd ~/openvla
eval "$(conda shell.bash hook)"
conda activate openvla

port=8000

/usr/bin/ssh -i ~/ssh/ssh_key -N -f -R $port:localhost:$port portal.gpu4edu.uvt.nl
python server_openvla.py
