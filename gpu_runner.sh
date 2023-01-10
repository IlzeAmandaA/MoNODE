#!/bin/bash
#SBATCH --gres=gpu:1              # Request one GPU
#SBATCH --time=7-00:00:00            # Runtime in hh:mm:ss
#SBATCH --mem=60G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --cpus-per-task=24        # cpus per task
#SBATCH --output=/home/iauzina/slurm_logs/%x_%j.out  # File to which STDOUT will be written
#SBATCH --error=/home/iauzina/slurm_logs/%x_%j.err   # File to which STDERR will be written
#SBATCH -D /home/iauzina/InvOdeVae

#include info about the job in the output
scontrol show job=$SLURM_JOB_ID

# conda activate virtual venv
source ~/.bashrc
conda deactivate
conda activate venv3.8
# Run Script
python main.py --Nepoch 110
