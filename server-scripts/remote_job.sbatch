#!/bin/bash
#SBATCH --job-name=cs4248_job
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Prefer nodes with Titan RTX GPUs (adjust if your cluster uses a different hostlist)
# This requests SLURM to place the job on one of xgpe0..xgpe11 where Titan RTX are available.
# If your cluster doesn't allow --nodelist in batch scripts, remove this line and instead
# pass a host preference when submitting (e.g. `sbatch --nodelist=xgpe[0-11] remote_job.sbatch`).
#SBATCH --nodelist=xgpe[0-11]

# Load modules or activate conda/venv here if required by your cluster
# module load cuda/11.8 python/3.10
# source /path/to/venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the project's remote run script (should be executable)
./remote_run.sh
