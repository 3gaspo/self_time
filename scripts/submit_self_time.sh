#!/bin/bash

set -euo pipefail

usage() {
    echo "usage: bash scripts/submit_self_time.sh dgx|selena" >&2
}

cluster="${1:-}"
case "$cluster" in
    dgx|selena) ;;
    *) usage; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
if [ "$cluster" = selena ]; then
    export PROJECT_ROOT
    source "$PROJECT_ROOT/src/slurm/selena_runtime.sh"
else
    TIME_STORAGE_ROOT="${TIME_STORAGE_ROOT:-$HOME}"
    export TIME_STORAGE_ROOT
    source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"
fi
mkdir -p "$TIME_LOGS/self_augmentation/slurm"
source "$PROJECT_ROOT/src/slurm/self_time_schedule.sh"

launch_id="${TIME_LAUNCH_ID:-${cluster}_$(date -u '+%Y%m%dT%H%M%SZ')_$$}"
jobs=()
for model in "${SELF_TIME_MODELS[@]}"; do
    if [ "$cluster" = selena ]; then
        front="$PROJECT_ROOT/slurm/selena/self_time/${model}_selena.slurm"
    else
        front="$PROJECT_ROOT/slurm/dgx/self_time/${model}.slurm"
    fi
    job_id="$(sbatch --parsable --export="ALL,TIME_LAUNCH_ID=$launch_id" "$front")"
    job_id="${job_id%%;*}"
    jobs+=("$job_id")
    echo "Self TIME submitted model=$model job_id=$job_id launch_id=$launch_id"
done

dependency="$(IFS=:; echo "${jobs[*]}")"
if [ "$cluster" = selena ]; then
    summary_front="$PROJECT_ROOT/slurm/selena/self_time/summary_selena.slurm"
else
    summary_front="$PROJECT_ROOT/slurm/dgx/self_time/summary.slurm"
fi
summary_job="$(sbatch --parsable --dependency="afterok:$dependency" \
    --export="ALL,TIME_LAUNCH_ID=$launch_id" "$summary_front")"
summary_job="${summary_job%%;*}"
echo "Self TIME summary submitted job_id=$summary_job dependency=afterok:$dependency"
echo "launch_id=$launch_id"
