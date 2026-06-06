#!/bin/bash
# ==============================================================================
# SCRIPT: cwn.sh
# DESCRIPTION:
#   Runs a scalable hyperparameter sweep for CWN (Cell Weisfeiler-Nahman
#   Network) across graph datasets lifted to cell complexes.
#   - ARCHITECTURE: Uses a "Cartesian Product" generation strategy.
#   - CONCURRENCY: Uses "Virtual Slots" to run N jobs per GPU.
#   - ORDERING: Prioritizes running all seeds for a config before moving on.
#   - FILTERING: Transductive datasets forced to batch_size=1.
# ==============================================================================

export SELECTED_GPUS="0,1,2,3,4,5,6,7" # Define your selected GPUs here
wandb_entity="" # Define your wandb entity here
RESUME=true  # Set to true to skip already-completed runs (reads SUCCESSFUL_RUNS.log)

# ==============================================================================
# SECTION 1: LOGGING & ENVIRONMENT SETUP
# ==============================================================================

# 1.1 Define Project Identifiers
script_name="$(basename "${BASH_SOURCE[0]}" .sh)"
project_name="${script_name}"
log_group="cwn_sweep"
LOG_DIR="./logs/${log_group}"

echo "=========================================================="
echo " Preparing log directory: $LOG_DIR"
echo "=========================================================="

# 1.2 Log directory management
if [[ "$RESUME" == "true" ]]; then
    echo "⏩ RESUME MODE: Keeping existing logs."
    mkdir -p "$LOG_DIR"
else
    if [ -d "$LOG_DIR" ]; then rm -r "$LOG_DIR"; fi
    mkdir -p "$LOG_DIR"
fi

# 1.3 Robust Dependency Loading
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export HYDRA_FULL_ERROR=1

find_logging_script() {
    local dir="$1"
    while [[ "$dir" != "/" ]]; do
        if [[ -f "$dir/base/logging.sh" ]]; then echo "$dir/base/logging.sh"; return 0; fi
        if [[ -f "$dir/scripts/base/logging.sh" ]]; then echo "$dir/scripts/base/logging.sh"; return 0; fi
        dir="$(dirname "$dir")"
    done
    return 1
}

LOGGING_PATH=$(find_logging_script "$SCRIPT_DIR")
if [[ -n "$LOGGING_PATH" ]]; then
    echo "✔ Found logging utils at: $LOGGING_PATH"
    source "$LOGGING_PATH"
else
    echo "❌ CRITICAL ERROR: Could not locate 'base/logging.sh'."
    exit 1
fi

# ==============================================================================
# CPU THREAD LIMITS (Crucial for concurrency)
# ==============================================================================
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ==============================================================================
# SECTION 2: HARDWARE & CONCURRENCY (Auto-Detected)
# ==============================================================================

# 2.1 Auto-detect GPUs and determine jobs-per-GPU from VRAM.
# Thresholds: >= 80 GB -> 5 jobs, <= 10 GB -> 1 job, <= 30 GB -> 2 jobs, else 3.
_gpu_info=$(python3 -c "
import subprocess
import os

selected_env = os.environ.get('SELECTED_GPUS', '').strip()
allowed_gpus = [x.strip() for x in selected_env.split(',')] if selected_env else None

try:
    out = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,memory.total', '--format=csv,noheader,nounits'],
        text=True
    )
    indices, mem_mb = [], []
    for line in out.strip().splitlines():
        idx, mem = line.split(',')
        idx = idx.strip()
        if allowed_gpus and idx not in allowed_gpus:
            continue
        indices.append(idx)
        mem_mb.append(int(mem.strip()))
    if not indices:
        print('0')
        exit(0)
    min_mem_gb = min(mem_mb) / 1024
    if min_mem_gb >= 80:
        jobs = 5
    elif min_mem_gb <= 10:
        jobs = 1
    elif min_mem_gb <= 30:
        jobs = 2
    else:
        jobs = 3
    print(jobs, ' '.join(indices))
except Exception:
    print('2 0')
")
read -r JOBS_PER_GPU _gpu_ids <<< "$_gpu_info"
read -ra physical_gpus <<< "$_gpu_ids"

echo "✔ Detected ${#physical_gpus[@]} GPU(s): ${physical_gpus[*]}"
echo "✔ Jobs per GPU: $JOBS_PER_GPU"

# 2.2 Create Virtual Slots
gpus=()
for gpu in "${physical_gpus[@]}"; do
    for ((i=1; i<=JOBS_PER_GPU; i++)); do gpus+=("$gpu"); done
done
echo "✔ Total virtual slots: ${#gpus[@]}"

# 2.3 Initialize Slot Tracking
declare -a slot_pids
for i in "${!gpus[@]}"; do slot_pids[$i]=0; done


# ==============================================================================
# SECTION 3: EXPERIMENT PARAMETERS
# ==============================================================================

# --- Model ---
# CWN is a cell-domain model; it is applied to graph datasets via graph2cell lifting.
models=(
    "cell/cwn"
)

# --- Datasets ---
# CWN requires cell-complex data. Only graph datasets are included here because
# they can be lifted to cell complexes (via cycle lifting). Simplicial datasets
# (mantra) cannot be lifted to cell complexes in this framework.
datasets=(
    "graph/MUTAG"
    "graph/PROTEINS"
    "graph/NCI1"
    "graph/NCI109"
    "graph/BBB_Martins"
    "graph/Caco2_Wang"
    "graph/Clearance_Hepatocyte_AZ"
    "graph/CYP3A4_Veith"
    "graph/cocitation_cora"
    "graph/cocitation_citeseer"
    "graph/cocitation_pubmed"
    "graph/ZINC"
)

# --- Hyperparameters ---
num_layers=(1 2 4)
hidden_channels=(64 128 256)
proj_dropouts=(0.0 0.25)
lrs=(0.01 0.001)
weight_decays=(0.0001)
batch_sizes=(128 256)
DATA_SEEDS=(0 3 5 7 9)

# --- Fixed Parameters ---
FIXED_ARGS=(
    "trainer.max_epochs=500"
    "trainer.min_epochs=50"
    "trainer.check_val_every_n_epoch=5"
    "callbacks.early_stopping.patience=10"
    "delete_checkpoint_after_test=True"
)


# ==============================================================================
# SECTION 4: SWEEP CONFIGURATION MAPPING
# Format: "ShortTag | HydraKey | ${Array[*]}"
# ==============================================================================

SWEEP_CONFIG=(
    # --- LEVEL 1: SLOWEST CHANGING (Outer Loops) ---
    "|model|${models[*]}"
    "|dataset|${datasets[*]}"

    # --- LEVEL 2: HYPERPARAMETERS ---
    "L|model.backbone.n_layers|${num_layers[*]}"
    "h|model.feature_encoder.out_channels|${hidden_channels[*]}"
    "pdro|model.feature_encoder.proj_dropout|${proj_dropouts[*]}"
    "lr|optimizer.parameters.lr|${lrs[*]}"
    "wd|optimizer.parameters.weight_decay|${weight_decays[*]}"
    "bs|dataset.dataloader_params.batch_size|${batch_sizes[*]}"

    # --- LEVEL 3: FASTEST CHANGING (Inner Loop) ---
    "seed|dataset.split_params.data_seed|${DATA_SEEDS[*]}"
)


# ==============================================================================
# SECTION 5: PYTHON GENERATOR (Transductive Filtering)
# ==============================================================================

export CONFIG_DIR="./configs/dataset"

generate_combinations() {
python3 -c "
import sys, itertools, os

config_dir = os.environ.get('CONFIG_DIR', './configs/dataset')

# 1. Parse Input Specs
specs = []
for item in sys.argv[1:]:
    parts = item.split('|')
    tag = parts[0].strip()
    key = parts[1].strip()
    vals = parts[2].split()
    specs.append({'tag': tag, 'key': key, 'vals': vals})

# 2. Generate Cartesian Product
options = [[(s['tag'], s['key'], val) for val in s['vals']] for s in specs]
combinations = list(itertools.product(*options))

# Helper to strip alias
def hydra_val(v):
    return v.split('::', 1)[1] if '::' in v else v

# Find the first batch size to avoid duplicating transductive runs
bs_key = 'dataset.dataloader_params.batch_size'
bs_spec = next((s for s in specs if s['key'] == bs_key), None)
first_bs = hydra_val(bs_spec['vals'][0]) if bs_spec else None

# 3. Filter and Mutate Combos
valid = []
skipped = 0
transductive_cache = {}

for combo in combinations:
    vals_dict = {key: hydra_val(val) for (_, key, val) in combo}
    dataset_val = vals_dict.get('dataset', '')
    current_bs = vals_dict.get(bs_key, '')

    # --- Transductive Batch Size Handler ---
    if dataset_val in transductive_cache:
        is_transductive = transductive_cache[dataset_val]
    else:
        is_transductive = False
        yaml_path = os.path.join(config_dir, f'{dataset_val}.yaml')
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                if 'learning_setting: transductive' in f.read():
                    is_transductive = True
        else:
            print(f'WARNING: Could not find config at {yaml_path}', file=sys.stderr)
        transductive_cache[dataset_val] = is_transductive

    if is_transductive:
        if current_bs != first_bs:
            skipped += 1
            continue
        new_combo = []
        for (tag, key, val) in combo:
            if key == bs_key:
                new_combo.append((tag, key, '1'))
            else:
                new_combo.append((tag, key, val))
        combo = tuple(new_combo)

    valid.append(combo)

# 4. Print header
print(f'TOTAL;{len(valid)}')
if skipped:
    print(f'SKIPPED;{skipped}', file=sys.stderr)

# 5. Print each valid combination
for combo in valid:
    name_parts = []
    cmd_args = []
    for (tag, key, val) in combo:
        if '::' in val:
            alias, actual_val = val.split('::', 1)
            clean_val = alias
        else:
            clean_val = os.path.basename(val)
            actual_val = val
        if tag:
            name_parts.append(f'{tag}{clean_val}')
        else:
            name_parts.append(clean_val)
        cmd_args.append(f'{key}={actual_val}')

    run_name = '_'.join(name_parts)
    print(f'{run_name};' + ' '.join(cmd_args))
" "${SWEEP_CONFIG[@]}"
}


# ==============================================================================
# SECTION 5.5: RESUME — LOAD COMPLETED RUNS
# ==============================================================================

declare -A _completed_runs
if [[ "$RESUME" == "true" ]]; then
    _success_log="$LOG_DIR/$log_group/SUCCESSFUL_RUNS.log"
    if [[ -f "$_success_log" ]]; then
        while IFS= read -r _line; do
            _rname="${_line##*\[SUCCESS\] }"
            _completed_runs["$_rname"]=1
        done < "$_success_log"
        echo "✔ Loaded ${#_completed_runs[@]} completed runs to skip."
    else
        echo "⚠️  No SUCCESSFUL_RUNS.log found at $_success_log — nothing to skip."
    fi
fi


# ==============================================================================
# SECTION 6: MAIN EXECUTION LOOP
# ==============================================================================

echo "----------------------------------------------------------"
echo " Generating experiment combinations..."
echo "----------------------------------------------------------"

total_runs=0
run_counter=0
skipped_completed=0
one_percent_step=1

while IFS=";" read -r col1 col2; do

    # 6.1 Handle Header
    if [[ "$col1" == "TOTAL" ]]; then
        total_runs=$col2
        if [ "$total_runs" -gt 0 ]; then
            one_percent_step=$(( total_runs / 100 ))
        fi
        if [ "$one_percent_step" -eq 0 ]; then one_percent_step=1; fi
        echo "► Total runs planned: $total_runs"
        echo "► Reporting progress every $one_percent_step runs (1%)"
        echo "----------------------------------------------------------"
        continue
    fi

    # 6.2 Parse Run Data
    run_name="$col1"
    dynamic_args_str="$col2"

    # 6.2.1 Skip if already completed (RESUME mode)
    if [[ "$RESUME" == "true" && -n "${_completed_runs[$run_name]+x}" ]]; then
        ((skipped_completed++))
        continue
    fi

    # 6.3 Update Progress
    ((run_counter++))
    if (( run_counter % one_percent_step == 0 )); then
        if [ "$total_runs" -gt 0 ]; then
            percent=$(( (run_counter * 100) / total_runs ))
        else
            percent=0
        fi
        echo "📊 Progress: ${percent}% completed ($run_counter / $total_runs runs launched)"
    fi

    # 6.4 Find a Free GPU Slot
    assigned_slot=-1
    while [ "$assigned_slot" -eq -1 ]; do
        for i in "${!gpus[@]}"; do
            pid="${slot_pids[$i]}"
            if [ "$pid" -eq 0 ] || ! kill -0 "$pid" 2>/dev/null; then
                assigned_slot=$i
                break
            fi
        done
        if [ "$assigned_slot" -eq -1 ]; then
            wait -n
        fi
    done

    # 6.5 Prepare Command
    current_gpu=${gpus[$assigned_slot]}
    read -ra DYNAMIC_ARGS_ARRAY <<< "$dynamic_args_str"

    # Extract dataset name for dynamic W&B project
    dataset_val=""
    for arg in "${DYNAMIC_ARGS_ARRAY[@]}"; do
        if [[ $arg == dataset=* ]]; then
            dataset_full_path="${arg#*=}"
            dataset_val=$(basename "$dataset_full_path")
            break
        fi
    done
    dynamic_project_name="${project_name}_${dataset_val}"

    cmd=(
        "python" "-m" "topobench"
        "${DYNAMIC_ARGS_ARRAY[@]}"
        "${FIXED_ARGS[@]}"
        "trainer.devices=[${current_gpu}]"
        "+logger.wandb.entity=${wandb_entity}"
        "logger.wandb.project=${dynamic_project_name}"
        "+logger.wandb.name=${run_name}"
    )

    # 6.6 Execute
    run_and_log "${cmd[*]}" "$log_group" "$run_name" "$LOG_DIR" &
    slot_pids[$assigned_slot]=$!

done < <(generate_combinations)


# ==============================================================================
# SECTION 7: CLEANUP
# ==============================================================================
echo "----------------------------------------------------------"
echo " All jobs launched ($run_counter total, $skipped_completed skipped as already completed)."
echo " Waiting for remaining background jobs to finish..."
echo "----------------------------------------------------------"
wait
