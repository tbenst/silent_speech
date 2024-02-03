#!/bin/bash
#SBATCH -p normal,owners,shauld,deissero
#SBATCH --job-name=interactive
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=120G

. activate magneto
cd /home/users/tbenst/code/silent_speech/notebooks/tyler
python 2023-07-25_dtw_speech_silent_emg.py