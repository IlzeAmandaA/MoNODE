#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=0-2:00            # Runtime in D-HH:MM
#SBATCH --mem=10000               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=/mnt/qb/work/bethge/cyildiz40/slurm_logs/%x_%j.out  # File to which STDOUT will be written
#SBATCH --error=/mnt/qb/work/bethge/cyildiz40/slurm_logs/%x_%j.err   # File to which STDERR will be written
#SBATCH --gres=gpu:1              # Request one GPU

# ssh -t cyildiz40@134.2.168.72 "cd /mnt/qb/work/bethge/cyildiz40/contrastive-continual-dynamics; squeue --user cyildiz40; conda activate default; bash -l"

# include information about the job in the output
scontrol show job=$SLURM_JOB_ID

# conda init bash
source activate default
# conda  activate default
srun python3 main.py
