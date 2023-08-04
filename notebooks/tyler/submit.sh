#!/bin/bash
#SBATCH -p normal,owners,shauld,deissero
#SBATCH --job-name=interactive
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --constraint=GPU_MEM:32GB

. activate magneto
cd /home/users/tbenst/code/silent_speech/notebooks/tyler
python 2023-07-25_dtw_speech_silent_emg.py