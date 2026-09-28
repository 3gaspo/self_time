#!/bin/bash

set -euo pipefail

usage() {
    echo "usage: bash scripts/self_time_status.sh dgx|selena [LAUNCH_ID]" >&2
}

cluster="${1:-}"
launch_id="${2:-}"
case "$cluster" in
    dgx|selena) ;;
    *) usage; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ "$cluster" = selena ]; then
    source "$PROJECT_ROOT/src/slurm/selena_runtime.sh"
else
    TIME_STORAGE_ROOT="${TIME_STORAGE_ROOT:-$HOME}"
    export TIME_STORAGE_ROOT
    source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"
fi

model_status_root="$TIME_LOGS/self_augmentation/workflow_status/self_time"
if [ -z "$launch_id" ]; then
    latest_status="$(ls -1t "$model_status_root"/*.status 2>/dev/null | head -n 1 || true)"
    if [ -z "$latest_status" ]; then
        echo "no Self TIME workflow status found below $model_status_root" >&2
        exit 1
    fi
    launch_id="$(sed -n 's/^launch_id=//p' "$latest_status")"
fi
launch_name="${launch_id//[^a-zA-Z0-9_.-]/_}"

echo "Self TIME status cluster=$cluster launch_id=$launch_id logs=$TIME_LOGS"
for workflow in self_time self_time_summary; do
    status_dir="$TIME_LOGS/self_augmentation/workflow_status/$workflow"
    [ -d "$status_dir" ] || continue
    for status_file in "$status_dir/${launch_name}__"*.status; do
        [ -f "$status_file" ] || continue
        task="$(sed -n 's/^task=//p' "$status_file")"
        state="$(sed -n 's/^state=//p' "$status_file")"
        stage="$(sed -n 's/^stage=//p' "$status_file")"
        launched="$(sed -n 's/^launched_at=//p' "$status_file")"
        updated="$(sed -n 's/^updated_at=//p' "$status_file")"
        printf '%-22s %-28s state=%-9s stage=%-12s launched=%s updated=%s\n' \
            "$workflow" "$task" "$state" "$stage" "$launched" "$updated"
    done
done

job_id="${launch_id##*_}"
if [[ "$job_id" =~ ^[0-9]+$ ]] && command -v squeue >/dev/null 2>&1; then
    echo
    squeue -j "$job_id" || true
fi
