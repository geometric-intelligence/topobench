#!/bin/bash
# ==============================================================================
# SCRIPT: hopse_preprocessing_time_ablation.sh
# DESCRIPTION:
#   Minimal training (2 epochs) to isolate **preprocessing / encoding time**
#   while sweeping **neighborhood structure** only — same 5 neighborhood presets
#   as hopse_m.sh / hopse_g.sh (adj1–adj3, inc1–inc2).
#
#   Intended Cartesian layout: 6 model branches × 12 datasets × 5 neighborhoods × 1 seed.
#   After the same **cell model + simplicial dataset** filter as hopse_m/hopse_g, the
#   generator emits **315** runs (45 invalid pairs removed: 3 simplicial datasets × 3
#   cell-only branches × 5 neighborhoods). Graph-only datasets keep all 6 branches.
#       - cell/hopse_m  + PE encodings (pse family)
#       - cell/hopse_m  + FE encodings (fe family)
#       - simplicial/hopse_m + PE
#       - simplicial/hopse_m + FE
#       - cell/hopse_g
#       - simplicial/hopse_g
#
#   Non-neighborhood hyperparameters are fixed to a single representative
#   point from the big sweeps. HOPSE-G uses one GPSE checkpoint:
#   transforms.hopse_encoding.pretrain_model=molpcba (override in FIXED_ARGS if needed).
# ==============================================================================
# DO NOT MISS THIS

export SELECTED_GPUS="0,1,2,3,4,5,6,7" # Define your selected GPUs here
wandb_entity="" # Define your wandb entity here
RESUME=false

# ==============================================================================
# SECTION 1: LOGGING & ENVIRONMENT SETUP
# ==============================================================================

trap 'echo -e "\n🛑 Interrupted! Cleaning up all background jobs..."; kill 0 2>/dev/null; exit 1' SIGINT SIGTERM

script_name="$(basename "${BASH_SOURCE[0]}" .sh)"
project_name="${script_name}"
log_group="hopse_preproc_time_ablation"
LOG_DIR="./logs/${log_group}"

echo "=========================================================="
echo " Preparing log directory: $LOG_DIR"
echo "=========================================================="

if [[ "$RESUME" == "true" ]]; then
    echo "⏩ RESUME MODE: Keeping existing logs."
    mkdir -p "$LOG_DIR"
else
    if [ -d "$LOG_DIR" ]; then rm -r "$LOG_DIR"; fi
    mkdir -p "$LOG_DIR"
fi

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

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export WANDB_START_METHOD="thread"
export WANDB__SERVICE_WAIT=300

# ==============================================================================
# SECTION 2: HARDWARE & CONCURRENCY (Auto-Detected)
# ==============================================================================

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
read -r _detected_jobs_per_gpu _gpu_ids <<< "$_gpu_info"
read -ra physical_gpus <<< "$_gpu_ids"

# Unlike hopse_m/hopse_g sweeps, this ablation must **never** stack multiple jobs on
# the same physical GPU (VRAM-based JOBS_PER_GPU is ignored here).
JOBS_PER_GPU=1

echo "✔ Detected ${#physical_gpus[@]} GPU(s): ${physical_gpus[*]}"
echo "✔ Jobs per GPU: $JOBS_PER_GPU (forced: one concurrent run per physical GPU)"

gpus=("${physical_gpus[@]}")
echo "✔ Total virtual slots: ${#gpus[@]} (= physical GPUs; one concurrent run per GPU)"

declare -a slot_pids
for i in "${!gpus[@]}"; do slot_pids[$i]=0; done

# ==============================================================================
# SECTION 3: FIXED TRAINING / WANDB (preprocessing-focused)
# ==============================================================================
# Two epochs only; high early-stopping patience so nothing stops early.
FIXED_ARGS=(
    "trainer.max_epochs=2"
    "trainer.min_epochs=1"
    "trainer.check_val_every_n_epoch=1"
    "callbacks.early_stopping.patience=999"
    "delete_checkpoint_after_test=True"
    "+combined_feature_encodings.preprocessor_device='cuda'"
)

# ==============================================================================
# SECTION 4: PYTHON GENERATOR (315 valid combos + transductive batch-size rule)
# ==============================================================================
export CONFIG_DIR="./configs/dataset"

generate_combinations() {
python3 -c "
import os
import sys

config_dir = os.environ.get('CONFIG_DIR', './configs/dataset')

# --- 6 model branches (PE/FE for HOPSE-M; plain HOPSE-G × cell/sim) ---
# (run_tag, model, kind, encodings_hydra_or_None)
# encodings match hopse_m.sh: pse:: / fe::
VARIANTS = [
    ('cell_m_pe', 'cell/hopse_m', 'hopse_m', '[LapPE,RWSE,ElectrostaticPE,HKdiagSE]'),
    ('cell_m_fe', 'cell/hopse_m', 'hopse_m', '[HKFE,KHopFE,PPRFE]'),
    ('sim_m_pe', 'simplicial/hopse_m', 'hopse_m', '[LapPE,RWSE,ElectrostaticPE,HKdiagSE]'),
    ('sim_m_fe', 'simplicial/hopse_m', 'hopse_m', '[HKFE,KHopFE,PPRFE]'),
    ('cell_g', 'cell/hopse_g', 'hopse_g', None),
    ('sim_g', 'simplicial/hopse_g', 'hopse_g', None),
]

# --- 12 datasets: hopse_g.sh active 8 + four common graph benchmarks from hopse_m comments ---
DATASETS = [
    'graph/BBB_Martins',
    'graph/Caco2_Wang',
    'graph/Clearance_Hepatocyte_AZ',
    'graph/CYP3A4_Veith',
    'simplicial/mantra_name',
    'simplicial/mantra_orientation',
    'simplicial/mantra_betti_numbers',
    'graph/ZINC',
    'graph/MUTAG',
    'graph/PROTEINS',
    'graph/NCI1',
    'graph/NCI109',
]

# --- Same 5 neighborhood presets as hopse_g.sh / hopse_m.sh (alias, hydra list) ---
NEIGHBORHOODS = [
    ('adj1', '[up_adjacency-0]'),
    ('adj2', '[up_adjacency-0,2-up_adjacency-0]'),
    ('adj3', '[up_adjacency-0,up_adjacency-1,2-up_adjacency-0,down_adjacency-1,down_adjacency-2,2-down_adjacency-2]'),
    ('inc1', '[up_incidence-0,2-up_incidence-0]'),
    ('inc2', '[up_incidence-0,up_incidence-1,2-up_incidence-0,down_incidence-1,down_incidence-2,2-down_incidence-2]'),
]

DATA_SEED = '0'
BS = '128'
PRETRAIN = 'molpcba'  # single GPSE checkpoint for all HOPSE-G rows

# One representative hyperparameter point from the big sweeps
FIXED_HP = [
    'model.backbone.n_layers=2',
    'model.feature_encoder.out_channels=128',
    'model.feature_encoder.proj_dropout=0.25',
    'optimizer.parameters.lr=0.01',
    'optimizer.parameters.weight_decay=0.0001',
    f'dataset.dataloader_params.batch_size={BS}',
    f'dataset.split_params.data_seed={DATA_SEED}',
]

transductive_cache = {}

def is_transductive(dataset_val: str) -> bool:
    if dataset_val in transductive_cache:
        return transductive_cache[dataset_val]
    yaml_path = os.path.join(config_dir, f'{dataset_val}.yaml')
    t = False
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r', encoding='utf-8', errors='ignore') as f:
            if 'learning_setting: transductive' in f.read():
                t = True
    else:
        print(f'⚠️ WARNING: Could not find config at {yaml_path}', file=sys.stderr)
    transductive_cache[dataset_val] = t
    return t

valid = []
skipped = 0

for run_tag, model, kind, enc in VARIANTS:
    for ds in DATASETS:
        if model.startswith('cell/') and ds.startswith('simplicial/'):
            skipped += len(NEIGHBORHOODS)
            continue
        bs_use = '1' if is_transductive(ds) else BS
        for nb_alias, nb_val in NEIGHBORHOODS:
            name_parts = [run_tag, ds.replace('/', '_'), f'N{nb_alias}', f'seed{DATA_SEED}']
            cmd_args = [
                f'model={model}',
                f'dataset={ds}',
                f'model.preprocessing_params.neighborhoods={nb_val}',
            ]
            if kind == 'hopse_m':
                cmd_args.append(f'model.preprocessing_params.encodings={enc}')
            else:
                cmd_args.append(f'transforms.hopse_encoding.pretrain_model={PRETRAIN}')
            cmd_args.extend(FIXED_HP)
            # Override batch size if transductive forced bs=1
            cmd_args = [a for a in cmd_args if not a.startswith('dataset.dataloader_params.batch_size=')]
            cmd_args.append(f'dataset.dataloader_params.batch_size={bs_use}')
            valid.append(('_'.join(name_parts), cmd_args))

print(f'TOTAL;{len(valid)}')
if skipped:
    print(f'SKIPPED_RUNS;{skipped}', file=sys.stderr)

for run_name, cmd_args in valid:
    print(run_name + ';' + ' '.join(cmd_args))
"
}

# ==============================================================================
# SECTION 5: RESUME — LOAD COMPLETED RUNS
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
echo " Generating preprocessing-time ablation combinations..."
echo "----------------------------------------------------------"

total_runs=0
run_counter=0
skipped_completed=0
one_percent_step=1

while IFS=";" read -r col1 col2; do

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

    run_name="$col1"
    dynamic_args_str="$col2"

    if [[ "$RESUME" == "true" && -n "${_completed_runs[$run_name]+x}" ]]; then
        ((skipped_completed++))
        continue
    fi

    ((run_counter++))
    if (( run_counter % one_percent_step == 0 )); then
        if [ "$total_runs" -gt 0 ]; then
            percent=$(( (run_counter * 100) / total_runs ))
        else
            percent=0
        fi
        echo "📊 Progress: ${percent}% completed ($run_counter / $total_runs runs launched)"
    fi

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

    current_gpu=${gpus[$assigned_slot]}
    read -ra DYNAMIC_ARGS_ARRAY <<< "$dynamic_args_str"

    # One W&B project for all runs; `run_name` encodes dataset / model / neighborhood.
    cmd=(
        "python" "-m" "topobench"
        "${DYNAMIC_ARGS_ARRAY[@]}"
        "${FIXED_ARGS[@]}"
        "trainer.devices=[${current_gpu}]"
        "+logger.wandb.entity=${wandb_entity}"
        "logger.wandb.project=${project_name}"
        "+logger.wandb.name=${run_name}"
    )

    run_and_log "${cmd[*]}" "$log_group" "$run_name" "$LOG_DIR" &
    slot_pids[$assigned_slot]=$!

done < <(generate_combinations)

echo "----------------------------------------------------------"
echo " All jobs launched ($run_counter total, $skipped_completed skipped as already completed)."
echo " Waiting for remaining background jobs to finish..."
echo "----------------------------------------------------------"
wait
echo "✔ All runs complete."
