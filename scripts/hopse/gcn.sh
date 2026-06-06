#!/bin/bash
# ==============================================================================
# SCRIPT: gcn.sh
# DESCRIPTION:
#   Runs a scalable hyperparameter sweep for HOPSE_M models across both
#   simplicial and cellular domains.
#   - ARCHITECTURE: Uses a "Cartesian Product" generation strategy.
#   - CONCURRENCY: Uses "Virtual Slots" to run N jobs per GPU.
#   - ORDERING: Prioritizes running all seeds for a config before moving on.
#   - FILTERING: Skips invalid model+dataset combos (cell + simplicial data).
# ==============================================================================

# ==============================================================================
# SECTION 1: LOGGING & ENVIRONMENT SETUP
# ==============================================================================

# 1.1 Define Project Identifiers
script_name="$(basename "${BASH_SOURCE[0]}" .sh)"
project_name="${script_name}"
log_group="gcn_sweep"
LOG_DIR="./logs/${log_group}"
wandb_entity="" # Define your wandb entity here

echo "=========================================================="
echo " Preparing log directory: $LOG_DIR"
echo "=========================================================="

# 1.2 Clean up old logs to ensure a fresh run
if [ -d "$LOG_DIR" ]; then rm -r "$LOG_DIR"; fi
mkdir -p "$LOG_DIR"

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
# SECTION 2: HARDWARE & CONCURRENCY (Auto-Detected)
# ==============================================================================

# 2.1 Auto-detect GPUs and determine jobs-per-GPU from VRAM.
# Output format: "JOBS_PER_GPU gpu_id_0 gpu_id_1 ..."
# Thresholds: >= 80 GB -> 4 jobs, <= 30 GB -> 2 jobs, between -> 3 jobs.
_gpu_info=$(python3 -c "
import subprocess
try:
    out = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,memory.total', '--format=csv,noheader,nounits'],
        text=True
    )
    indices, mem_mb = [], []
    for line in out.strip().splitlines():
        idx, mem = line.split(',')
        indices.append(idx.strip())
        mem_mb.append(int(mem.strip()))
    min_mem_gb = min(mem_mb) / 1024
    if min_mem_gb >= 80:
        jobs = 4
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

# --- Models ---
models=(
    "gcn::graph/gcn"
)

# --- Datasets ---
datasets=(
    "graph/MUTAG"
    "graph/cocitation_cora"
    "graph/PROTEINS"
    "graph/NCI1"
    "graph/NCI109"
    "graph/ZINC"
    "graph/cocitation_citeseer"
    "graph/cocitation_pubmed"
    "simplicial/mantra_name"
    "simplicial/mantra_orientation"
    "simplicial/mantra_betti_numbers"
)

# --- Transforms (Hydra: configs/transforms/<name>.yaml) ---
# combined_pe / combined_fe nest data_manipulations under CombinedPSEs / CombinedFEs (encoding lists in those YAMLs).
# Extra Hydra flags: @@@ between full key=value pieces, e.g.
#   "pse::combined_pe@@@transforms.CombinedPSEs.encodings=[LapPE,RWSE]"
transform_presets=(
    "notf::no_transform"
    "pse::combined_pe@@@transforms.CombinedPSEs.encodings=[LapPE,RWSE,ElectrostaticPE,HKdiagSE]"
    "fe::combined_fe@@@transforms.CombinedFEs.encodings=[HKFE,KHopFE,PPRFE]"
)

# --- Hyperparameters (superset across all dataset groups) ---
num_layers=(1 2 4)
hidden_channels=(128 256)
proj_dropouts=(0.25 0.5)
lrs=(0.01 0.001)
weight_decays=(0 0.0001)
batch_sizes=(128 256)
DATA_SEEDS=(0 3 5 7 9)

# --- Fixed Parameters ---
FIXED_ARGS=(
    "trainer.max_epochs=500"
    "trainer.min_epochs=50"
    "trainer.check_val_every_n_epoch=5"
    "callbacks.early_stopping.patience=5"
)


# ==============================================================================
# SECTION 4: SWEEP CONFIGURATION MAPPING (CRITICAL ORDERING)
# Format: "ShortTag | HydraKey | ${Array[*]}"
#
# Values support an optional "alias::hydra_value" syntax for readable names.
# Use @@@ in hydra_value to emit several space-separated CLI overrides (see transform_presets).
# The generator also filters out invalid model+dataset combos.
# ==============================================================================

SWEEP_CONFIG=(
    # --- LEVEL 1: SLOWEST CHANGING (Outer Loops) ---
    "|model|${models[*]}"
    "|dataset|${datasets[*]}"
    "tf|transforms|${transform_presets[*]}"

    # --- LEVEL 2: HYPERPARAMETERS ---
    "L|model.backbone.num_layers|${num_layers[*]}"
    "h|model.feature_encoder.out_channels|${hidden_channels[*]}"
    "pdro|model.feature_encoder.proj_dropout|${proj_dropouts[*]}"
    "lr|optimizer.parameters.lr|${lrs[*]}"
    "wd|optimizer.parameters.weight_decay|${weight_decays[*]}"
    "bs|dataset.dataloader_params.batch_size|${batch_sizes[*]}"

    # --- LEVEL 3: FASTEST CHANGING (Inner Loop) ---
    "seed|dataset.split_params.data_seed|${DATA_SEEDS[*]}"
)


# ==============================================================================
# SECTION 5: PYTHON GENERATOR (Smart Transductive Filtering)
# ==============================================================================

# Define where your dataset YAMLs live so the generator can inspect them.
# UPDATE THIS PATH IF YOUR CONFIGS ARE STORED ELSEWHERE.
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

# Find the first batch size in the sweep so we don't duplicate transductive runs
bs_key = 'dataset.dataloader_params.batch_size'
bs_spec = next((s for s in specs if s['key'] == bs_key), None)
first_bs = hydra_val(bs_spec['vals'][0]) if bs_spec else None

# 3. Filter and Mutate Combos
valid = []
skipped = 0
transductive_cache = {}

for combo in combinations:
    vals_dict = {key: hydra_val(val) for (_, key, val) in combo}
    model_val = vals_dict.get('model', '')
    dataset_val = vals_dict.get('dataset', '')
    current_bs = vals_dict.get(bs_key, '')

    # --- Rule A: Skip cell model + simplicial dataset ---
    if model_val.startswith('cell/') and dataset_val.startswith('simplicial/'):
        skipped += 1
        continue

    # --- Rule B: Transductive Batch Size Handler ---
    is_transductive = False
    if dataset_val in transductive_cache:
        is_transductive = transductive_cache[dataset_val]
    else:
        # Construct path to yaml (e.g., ./configs/dataset/graph/cocitation_cora.yaml)
        yaml_path = os.path.join(config_dir, f'{dataset_val}.yaml')
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                # Fast text check avoids needing pip install pyyaml
                if 'learning_setting: transductive' in f.read():
                    is_transductive = True
        else:
            print(f'⚠️ WARNING: Could not find config at {yaml_path}', file=sys.stderr)

        transductive_cache[dataset_val] = is_transductive

    if is_transductive:
        # If this isn't the first batch size in the sweep list, skip it
        # to avoid running the exact same bs=1 experiment multiple times.
        if current_bs != first_bs:
            skipped += 1
            continue

        # Mutate the current combination to force batch_size to 1
        new_combo = []
        for (tag, key, val) in combo:
            if key == bs_key:
                # Force the value to 1. If an alias was used, keep it clean.
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
            alias, hydra_val_str = val.split('::', 1)
            clean_val = alias
            actual_val = hydra_val_str
        else:
            clean_val = os.path.basename(val)
            actual_val = val

        if tag:
            name_parts.append(f'{tag}{clean_val}')
        else:
            name_parts.append(clean_val)
        if '@@@' in actual_val:
            for part in actual_val.split('@@@'):
                part = part.strip()
                if part:
                    cmd_args.append(part)
        else:
            cmd_args.append(f'{key}={actual_val}')

    run_name = '_'.join(name_parts)
    print(f'{run_name};' + ' '.join(cmd_args))
" "${SWEEP_CONFIG[@]}"
}

# ==============================================================================
# SECTION 6: MAIN EXECUTION LOOP
# ==============================================================================

# If IFS was polluted, read can split transforms=combined_pe; Hydra then errors on bare "combined_pe".
repair_hydra_transforms_arg() {
    local -n _r=$1
    local out=() i
    for ((i = 0; i < ${#_r[@]}; i++)); do
        local t="${_r[i]}"
        if [[ "$t" == transforms=* ]]; then
            out+=("$t")
        elif [[ "$t" == "transforms" && $((i + 1)) -lt ${#_r[@]} ]]; then
            local nxt="${_r[$((i + 1))]}"
            [[ "$nxt" == *"="* ]] && { out+=("$t"); continue; }
            out+=("transforms=$nxt")
            ((i++))
        elif [[ "$t" =~ ^(combined_pe|combined_fe|no_transform)$ ]]; then
            out+=("transforms=$t")
        else
            out+=("$t")
        fi
    done
    _r=("${out[@]}")
}

echo "----------------------------------------------------------"
echo " Generating experiment combinations..."
echo "----------------------------------------------------------"

total_runs=0
run_counter=0
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
    # Must not use inherited IFS (e.g. IFS== splits transforms=combined_pe → bare "combined_pe" for Hydra)
    IFS=$' \t\n' read -ra DYNAMIC_ARGS_ARRAY <<< "$dynamic_args_str"
    repair_hydra_transforms_arg DYNAMIC_ARGS_ARRAY

    # --- Extract dataset name for dynamic W&B project ---
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
    )

    # 6.6 Execute — printf %q so run_and_log's eval keeps key=value overrides as single words
    # (broken IFS or nullglob in the parent shell can otherwise split transforms=combined_pe, etc.)
    cmd_eval=$(printf '%q ' "${cmd[@]}")
    run_and_log "${cmd_eval% }" "$log_group" "$run_name" "$LOG_DIR" &
    slot_pids[$assigned_slot]=$!

done < <(generate_combinations)


# ==============================================================================
# SECTION 7: CLEANUP
# ==============================================================================
echo "----------------------------------------------------------"
echo " All jobs launched ($run_counter total)."
echo " Waiting for remaining background jobs to finish..."
echo "----------------------------------------------------------"
wait
echo "✔ All runs complete."
