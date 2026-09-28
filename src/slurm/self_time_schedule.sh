#!/bin/bash

SELF_TIME_MODELS=(chronos2 ts_icl)

require_self_time_model() {
    local requested="$1"
    local candidate
    for candidate in "${SELF_TIME_MODELS[@]}"; do
        [ "$candidate" = "$requested" ] && return 0
    done
    echo "unknown Self TIME model=$requested; expected: ${SELF_TIME_MODELS[*]}" >&2
    return 2
}
