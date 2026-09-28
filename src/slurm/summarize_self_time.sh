#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must be set by the Slurm front}"
source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"

TIME_EXPERIMENT=self_augmentation
TIME_WORKFLOW_NAME=self_time_summary
TIME_TASK_NAME=mase_timing_and_heatmaps
TIME_STATUS_NAME=summary
TIME_LAUNCH_ID="${TIME_LAUNCH_ID:-${SLURM_JOB_ID:-manual_$(date -u '+%Y%m%dT%H%M%SZ')_$$}}"
export TIME_EXPERIMENT TIME_WORKFLOW_NAME TIME_TASK_NAME TIME_STATUS_NAME TIME_LAUNCH_ID
source "$PROJECT_ROOT/src/slurm/workflow_common.sh"

time_workflow_init
time_stage_start report
report_root="$TIME_OUTPUTS/self_augmentation/reports"
time_task_start "results=$TIME_OUTPUTS/self_augmentation/tasks seasonal=$TIME_SEASONAL_TASKS_ROOT output=$report_root"
command=(
    uv run --no-sync python "$PROJECT_ROOT/src/scripts/report.py"
    --results-dir "$TIME_OUTPUTS/self_augmentation/tasks"
    --seasonal-naive-results-dir "$TIME_SEASONAL_TASKS_ROOT"
    --launch-id "$TIME_LAUNCH_ID"
    --output "$report_root"
)
if [ -n "${SLURM_JOB_ID:-}" ]; then
    srun --ntasks=1 "${command[@]}"
else
    "${command[@]}"
fi
time_task_complete
time_stage_complete
time_workflow_complete
