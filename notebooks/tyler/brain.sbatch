#!/usr/bin/bash
#SBATCH --output=z%j.out
#SBATCH --error=z%j.err
#SBATCH --time=48:00:00
#SBATCH -p deissero
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH  --gpus=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem=64GB

# --gpus N, --ntasks-per-node N, and --devices N must all be the same value
# TODO: auto wal/ql-time resubmittion:
# https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html#enable-auto-wall-time-resubmitions
. activate magneto
ml nccl/2.17.1
cd ~/code/silent_speech/notebooks/tyler/
srun python 2023-08-16_brain_to_text.py --learning-rate="6e-4" --devices=2