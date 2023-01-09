#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --gres=gpu:1              # Request one GPU
#SBATCH --time=8:00:00            # Runtime in hh:mm:ss
#SBATCH --mem=30G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --cpus-per-task=10        # cpus per task
#SBATCH --output=/home/iauzina/slurm_logs/%x_%j.out  # File to which STDOUT will be written
#SBATCH --error=/home/iazuina/slurm_logs/%x_%j.err   # File to which STDERR will be written


# include information about the job in the output
scontrol show job=$SLURM_JOB_ID

# conda init bash
source activate venv3.8
# conda  activate default
srun python main.py --Nepoch 100
