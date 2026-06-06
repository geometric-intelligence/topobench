#!/bin/bash

# ==========================================================
# 🌎 GLOBAL LOGGING CONFIGURATION
# ==========================================================
# This variable sets the default root directory for all logs.
#
# The ":=" syntax means:
# If $LOGGING_ROOT_DIR is already set (e.g., by an calling script), use that value.
# If $LOGGING_ROOT_DIR is unset or null, set it to the default value "./logs".
#
: "${LOGGING_ROOT_DIR:=./logs}"


# ==========================================================
# 🚀 LOGGING ENGINE
# ==========================================================

# ---
# Main function to execute and log a command.
# It automatically handles success/failure logging, directory creation,
# and provides live output to the console.
#
# All logs (summary and detailed) are saved within the specific log_group directory.
#
# Usage:
#   run_and_log "command_to_run" "log_group_name" "run_name"
#
# Arguments:
#   $1: cmd       (string) The full command to execute
#   $2: log_group (string) The subdirectory for this group of experiments (e.g., "sweep_modularity")
#   $3: run_name  (string) The unique name for this specific run (e.g., "modelA_lr0.01")
#   $4: root_dir  (string) [OPTIONAL] A one-time override for the log root.
#                          Defaults to $LOGGING_ROOT_DIR.
# ---
run_and_log() {
    local cmd="$1"
    local log_group="$2"
    local run_name="$3"
    local root_dir="${4:-$LOGGING_ROOT_DIR}"

    local specific_log_dir="$root_dir/$log_group"
    mkdir -p "$specific_log_dir"

    local success_log="$specific_log_dir/SUCCESSFUL_RUNS.log"
    local failed_log="$specific_log_dir/FAILED_RUNS.log"

    local stdout_log="$specific_log_dir/${run_name}_stdout.log"
    local stderr_log="$specific_log_dir/${run_name}_stderr.log"

    # Unique temp files using the background subshell PID
    local tmp_stdout="${stdout_log}.${BASHPID}.tmp"
    local tmp_stderr="${stderr_log}.${BASHPID}.tmp"

    # --- RETRY CONFIGURATION ---
    local max_attempts=2
    local wait_time=15 # Seconds to wait before retrying
    local exit_code=0

    echo "--- [START] Running: $run_name (PID: $BASHPID) ---"

    for attempt in $(seq 1 $max_attempts); do

        # If this is a retry, print a warning and sleep
        if [ "$attempt" -gt 1 ]; then
            echo "⚠️ [RETRY] $run_name failed on attempt $((attempt-1)). Waiting ${wait_time}s before attempt $attempt..."
            sleep $wait_time
        fi

        # Execute synchronously.
        # Note: > overwrites the temp file on a retry, so you only keep the logs of the current attempt.
        eval "$cmd" > "$tmp_stdout" 2> "$tmp_stderr"
        exit_code=$?

        # If it succeeds, break out of the retry loop early
        if [ $exit_code -eq 0 ]; then
            break
        fi
    done

    # --- FINAL LOGGING ---
    if [ $exit_code -eq 0 ]; then
        echo "✅ [SUCCESS] Finished: $run_name"

        # Safe concurrent writing for the success log
        (
            flock -x 200
            echo "$(date): [SUCCESS] ${run_name}" >> "$success_log"
        ) 200> "${specific_log_dir}/.success.lock"

        # Clean up temp files
        rm -f "$tmp_stdout" "$tmp_stderr"
        return 0
    else
        echo "❌ [FAILURE] Finished: $run_name (Failed after $max_attempts attempts. Exit Code: $exit_code)"

        # Move the temp files from the final failed attempt to permanent logs
        mv "$tmp_stdout" "$stdout_log"
        mv "$tmp_stderr" "$stderr_log"

        # Safe concurrent writing for the failure log
        (
            flock -x 200
            echo "=================================" >> "$failed_log"
            echo "FAILURE on $(date): [${run_name}]" >> "$failed_log"
            echo "Exit Code: $exit_code" >> "$failed_log"
            echo "Attempts: $max_attempts" >> "$failed_log"
            echo "Command: $cmd" >> "$failed_log"
            echo "See full logs: $stdout_log | $stderr_log" >> "$failed_log"
            echo "=================================" >> "$failed_log"
        ) 200> "${specific_log_dir}/.failed.lock"

        # Print the error to the console so you can still monitor failures live
        echo "----------------- ERROR OUTPUT ($run_name) -----------------"
        tail -n 15 "$stderr_log"
        echo "----------------------------------------------------------------"

        return 1
    fi
}


# Example:

#!/bin/bash

# 1. Source the updated logging utility
# (Adjust path if run_utils.sh is in a different directory)
# source ./run_utils.sh

# # 2. Define a single root directory for all logs
# ROOT_LOG_DIR="scripts/checks/experiment_logs"

# # 3. Clear previous *global summary* log files
# mkdir -p "$ROOT_LOG_DIR"
# > "$ROOT_LOG_DIR/SUCCESSFUL_RUNS.log"
# > "$ROOT_LOG_DIR/FAILED_RUNS.log"


# MODELS=("stf" "patchtst")
# # "lstm"  "stf" "patchtst" "cnn_1d"  "stf" "patchtst"

# # 4. Define command templates in an array
# # Use SINGLE QUOTES (') to prevent ${model} from expanding now
# commands=(
#     'python lmetk/run.py experiment=xjtu_sy/prognostics/spectral/${model}.yaml trainer.max_epochs=100 trainer.min_epochs=50 optimizer.parameters.lr=0.01 task_definition.seq_len=16'
#     'python lmetk/run.py experiment=pronostia/prognostics/spectral/${model}.yaml trainer.max_epochs=100 trainer.min_epochs=50 optimizer.parameters.lr=0.01 task_definition.seq_len=16'
#     'python lmetk/run.py experiment=pronostia/prognostics/spectral/${model}.yaml trainer.max_epochs=100 trainer.min_epochs=50 optimizer.parameters.lr=0.001 task_definition.seq_len=16'
#     # Add other commands here
# )

# # 5. Define corresponding log groups for each command
# # This array MUST match the 'commands' array element-for-element
# log_groups=(
#     "xjtu_sy/spectral/lr_0.01"
#     "pronostia/spectral/lr_0.01"
#     "pronostia/spectral/lr_0.001"
#     # Add corresponding log group names here
# )

# # 6. Loop using array indices to keep commands and log groups in sync
# for i in "${!commands[@]}"; do
#     cmd_template="${commands[$i]}"
#     log_group="${log_groups[$i]}"

#     echo "============================================================"
#     echo "Starting Experiment Group: $log_group"
#     echo "============================================================"

#     for model in "${MODELS[@]}"; do

#         # 7. Use 'eval' to substitute the $model variable into the command
#         eval "final_cmd=\"$cmd_template\""

#         # 8. Define a unique run name for this specific job.
#         # Here, we just use the model name.
#         local run_name="$model"

#         # 9. Call the new, general-purpose logger
#         # It will handle all logging, directory creation, and error reporting.
#         run_and_log "$final_cmd" "$log_group" "$run_name" "$ROOT_LOG_DIR"

#         # The return code (0 for success, 1 for failure) is available
#         # in $? if you need to check it, e.g., to stop the script.
#         # if [ $? -ne 0 ]; then
#         #    echo "!!! Critical error on $model, stopping script."
#         #    exit 1
#         # fi
#     done
# done

# echo "All experiments finished."
