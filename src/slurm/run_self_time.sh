#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must be set by the Slurm front}"
source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"
source "$PROJECT_ROOT/src/slurm/self_time_schedule.sh"

model="${TIME_MODEL:?TIME_MODEL must be chronos2 or ts_icl}"
require_self_time_model "$model"

TIME_EXPERIMENT=self_augmentation
TIME_WORKFLOW_NAME=self_time
TIME_TASK_NAME="$model"
TIME_STATUS_NAME="$model"
TIME_LAUNCH_ID="${TIME_LAUNCH_ID:-${SLURM_JOB_ID:-manual_$(date -u '+%Y%m%dT%H%M%SZ')_$$}}"
TIME_RESULT_SCOPE="$TIME_OUTPUTS/self_augmentation/tasks/$model"
export TIME_EXPERIMENT TIME_WORKFLOW_NAME TIME_TASK_NAME TIME_STATUS_NAME
export TIME_LAUNCH_ID TIME_RESULT_SCOPE
source "$PROJECT_ROOT/src/slurm/workflow_common.sh"

time_workflow_init
time_stage_start evaluate
time_task_start "model=$model config=src/conf/self_time.yaml datasets=all_datasets covariates=past_only instance_normalization=zscore"
command=(
    uv run --no-sync python "$PROJECT_ROOT/src/scripts/run_self_time.py"
    "model=$model"
)
if [ -n "${SLURM_JOB_ID:-}" ]; then
    srun --ntasks=1 "${command[@]}"
else
    "${command[@]}"
fi
time_task_complete
time_stage_complete
time_workflow_complete
