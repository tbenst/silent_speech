#!/bin/bash
#SBATCH -p henderj
#SBATCH --job-name=interactive
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --constraint=GPU_MEM:80GB

. activate magneto
cd /home/users/tbenst/code/silent_speech/notebooks/tyler
python 2024-01-15_icml_models.py