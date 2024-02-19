#!/bin/bash
#SBATCH -p henderj,deissero,shauld
#SBATCH --job-name=interactive
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G

. activate magneto
cd /home/users/tbenst/code/silent_speech/notebooks/tyler
python 2023-07-17_cache_dataset_with_attrs_.py