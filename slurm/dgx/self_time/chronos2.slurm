#!/bin/bash
#SBATCH --job-name=selfc2
#SBATCH --gres=gpu:1
#SBATCH --output=logs/dgx/self_augmentation/slurm/%x_%j.out
#SBATCH --error=logs/dgx/self_augmentation/slurm/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80000

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT TIME_MODEL=chronos2
export TIME_LAUNCH_ID="${TIME_LAUNCH_ID:-dgx_${SLURM_JOB_ID}}"
source "$PROJECT_ROOT/src/slurm/run_self_time.sh"
