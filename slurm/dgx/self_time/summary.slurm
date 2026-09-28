#!/bin/bash
#SBATCH --job-name=selfsum
#SBATCH --gres=gpu:1
#SBATCH --output=logs/dgx/self_augmentation/slurm/%x_%j.out
#SBATCH --error=logs/dgx/self_augmentation/slurm/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT
export TIME_LAUNCH_ID="${TIME_LAUNCH_ID:-dgx_${SLURM_JOB_ID}}"
source "$PROJECT_ROOT/src/slurm/summarize_self_time.sh"
