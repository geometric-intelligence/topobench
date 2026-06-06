#!/usr/bin/env python3
"""
Download all W&B runs for a specific entity and projects starting with a prefix.
Saves each project's runs as a CSV file in tutorials/wandb_runs/.
Aggregates results by hyperparameters, averaging over different data seeds.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import wandb
from tqdm import tqdm

WANDB_ENTITY = "gbg141-hopse"
PROJECT_PREFIX = "topotune"
OUTPUT_DIR = Path("tutorials/wandb_runs/")
AGGREGATED_FILE = OUTPUT_DIR / "aggregated_results.csv"

# Hyperparameters to group by
GROUP_HPARAMS = [
    "config.model.model_domain",
    "config.model.tune_gnn",
    "config.model.backbone.neighborhoods",
    "config.model.backbone.GNN.num_layers",
    "config.model.feature_encoder.out_channels",
    "config.model.feature_encoder.proj_dropout",
    "config.optimizer.parameters.lr",
    "config.optimizer.parameters.weight_decay",
    "config.dataset.dataloader_params.batch_size",
]

def flatten_config(obj: Any, parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flattens a nested dictionary, similar to the one in scripts/hopse_plotting/utils.py"""
    out: dict[str, Any] = {}
    if not isinstance(obj, Mapping):
        return {parent_key: obj} if parent_key else {}

    for k, v in obj.items():
        k = str(k)
        if not parent_key and k.startswith("_"):
            continue
        
        # Handle W&B config format which often has {'value': ..., 'desc': ...}
        if isinstance(v, dict) and "value" in v:
            v = v["value"]
            
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, Mapping):
            out.update(flatten_config(v, new_key, sep=sep))
        else:
            out[new_key] = v
    return out

def download_project_runs(api: wandb.Api, entity: str, project: str, output_path: Path, force: bool = False):
    """Downloads all runs for a given project and saves them to a CSV."""
    if output_path.exists() and not force:
        print(f"Skipping download for {entity}/{project} (file already exists: {output_path})")
        return

    print(f"Downloading runs for project: {entity}/{project}")
    runs = api.runs(f"{entity}/{project}")
    
    rows = []
    for run in tqdm(runs, desc=f"Runs in {project}", leave=False):
        # Basic metadata
        run_data = {
            "run_id": run.id,
            "run_name": run.name,
            "run_state": run.state,
            "run_url": run.url,
            "run_tags": ",".join(run.tags) if run.tags else "",
            "run_created_at": run.created_at,
        }
        
        # Flatten config
        config = flatten_config(run.config)
        for k, v in config.items():
            run_data[f"config.{k}"] = v
            
        # Flatten summary (metrics)
        summary = flatten_config(run.summary._json_dict)
        for k, v in summary.items():
            run_data[f"summary.{k}"] = v
            
        rows.append(run_data)
    
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"  Saved {len(rows)} runs to {output_path}")
    else:
        print(f"  No runs found for project {project}")

def aggregate_results(input_dir: Path, output_file: Path):
    """Aggregates all CSVs in input_dir into a single summary CSV."""
    csv_files = list(input_dir.glob(f"{PROJECT_PREFIX}_*.csv"))
    if not csv_files:
        print("No CSV files found for aggregation.")
        return

    all_dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        # Extract dataset name from project filename
        dataset_name = f.stem.replace(f"{PROJECT_PREFIX}_", "")
        df["dataset_name"] = dataset_name
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Ensure all group hparams exist in the dataframe
    existing_group_cols = [c for c in GROUP_HPARAMS if c in combined_df.columns]
    missing_group_cols = set(GROUP_HPARAMS) - set(existing_group_cols)
    if missing_group_cols:
        print(f"Warning: Missing group columns in data: {missing_group_cols}")

    group_cols = ["dataset_name"] + existing_group_cols

    # Dedup: If multiple runs exist for the same hyperparameters AND same seed,
    # keep only the latest one (highest created_at or just last in list).
    if "config.dataset.split_params.data_seed" in combined_df.columns:
        print("Deduplicating runs with same hyperparameters and seed (keeping latest)...")
        initial_count = len(combined_df)
        combined_df = combined_df.sort_values("run_created_at", ascending=False)
        dedup_cols = group_cols + ["config.dataset.split_params.data_seed"]
        combined_df = combined_df.drop_duplicates(subset=dedup_cols, keep="first")
        final_count = len(combined_df)
        print(f"  Removed {initial_count - final_count} duplicate runs.")

    # Debug: Check for varying hyperparameters within groups
    print("\n--- Debug: Varying hyperparameters within groups (excluding seed) ---")
    config_cols = [c for c in combined_df.columns if c.startswith("config.") and c not in GROUP_HPARAMS and "data_seed" not in c]
    
    # Check all groups for varying configs to be sure
    varying_found = False
    for name, group in combined_df.groupby(group_cols):
        varying_in_group = {}
        for col in config_cols:
            unique_vals = group[col].unique()
            if len(unique_vals) > 1:
                # Filter out nans if they are causing noise, but here we want to see everything
                varying_in_group[col] = unique_vals.tolist()
        
        if varying_in_group:
            varying_found = True
            print(f"Group {name} has varying configs:")
            for col, vals in varying_in_group.items():
                print(f"  - {col}: {vals}")
            # Limit output to first few groups with issues to avoid spam
            # but we show one from each dataset if possible
            if len(name) > 0 and name[0] == "BBB_Martins": 
                # keep going for a bit
                pass
            else:
                break
    
    if not varying_found:
        print("No unexpected varying config columns found within groups.")
    print("----------------------------------------------------\n")

    # Filter for target columns
    metric_cols = [c for c in combined_df.columns if c.startswith("summary.test_best_rerun/") or c.startswith("summary.val_best_rerun/")]

    # Check for metrics
    if not metric_cols:
        print("No 'summary.*_best_rerun/*' metrics found to aggregate.")
        return

    # Convert metrics to numeric, handling potential issues
    for col in metric_cols:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

    # Aggregation
    agg_dict = {col: ['mean', 'std'] for col in metric_cols}
    
    # Add count
    summary_df = combined_df.groupby(group_cols).agg(agg_dict)
    
    # Flatten multi-index columns
    summary_df.columns = [f"{col}_{stat}" for col, stat in summary_df.columns]
    
    # Add count column
    summary_df["run_count"] = combined_df.groupby(group_cols).size()
    
    summary_df = summary_df.reset_index()
    
    summary_df.to_csv(output_file, index=False)
    print(f"Aggregated results saved to {output_file}")

def save_best_runs(aggregated_file: Path, output_file: Path):
    """Saves the best run (by validation performance) for each dataset/domain."""
    if not aggregated_file.exists():
        return
    
    df = pd.read_csv(aggregated_file)
    
    # Filter out cocitation and ZINC
    df = df[~df["dataset_name"].str.contains("cocitation", case=False)]
    df = df[~df["dataset_name"].str.contains("ZINC", case=False)]
    
    processed_rows = []
    domain_col = "config.model.model_domain"
    
    for ds_name in df["dataset_name"].unique():
        ds_full_group = df[df["dataset_name"] == ds_name].copy()
        domains = ds_full_group[domain_col].unique()
        
        for domain in domains:
            ds_group = ds_full_group[ds_full_group[domain_col] == domain].copy()
            
            if "betti_numbers" in ds_name:
                for suffix in ["1", "2"]:
                    metric_base = f"f1-{suffix}"
                    # Select based on VAL
                    val_mean_col = f"summary.val_best_rerun/{metric_base}_mean"
                    # Report TEST
                    test_mean_col = f"summary.test_best_rerun/{metric_base}_mean"
                    test_std_col = f"summary.test_best_rerun/{metric_base}_std"
                    
                    if val_mean_col not in ds_group.columns:
                        continue
                        
                    sub_ds = ds_group.dropna(subset=[val_mean_col]).copy()
                    if sub_ds.empty:
                        continue
                    
                    # Best is max val f1
                    best_idx = sub_ds[val_mean_col].idxmax()
                    best_row = sub_ds.loc[best_idx].copy()
                    
                    best_row["dataset_name"] = f"mantra_betti_number_{suffix}"
                    best_row["score_name"] = "f1"
                    best_row["score"] = best_row[test_mean_col]
                    best_row["score_std"] = best_row[test_std_col]
                    best_row["val_score"] = best_row[val_mean_col]
                    processed_rows.append(best_row)
            else:
                if ds_name in ["Clearance_Hepatocyte_AZ", "Caco2_Wang"]:
                    # Regression: Select by min val MAE, report test MAE
                    ds_group["score_name"] = "mae"
                    val_col = "summary.val_best_rerun/mae_mean"
                    test_col = "summary.test_best_rerun/mae_mean"
                    test_std = "summary.test_best_rerun/mae_std"
                    
                    sub_ds = ds_group.dropna(subset=[val_col]).copy()
                    if sub_ds.empty: continue
                    best_idx = sub_ds[val_col].idxmin()
                elif "mantra" in ds_name:
                    # Mantra: Select by max val F1, report test F1
                    ds_group["score_name"] = "f1"
                    val_col = "summary.val_best_rerun/f1_mean"
                    test_col = "summary.test_best_rerun/f1_mean"
                    test_std = "summary.test_best_rerun/f1_std"
                    
                    sub_ds = ds_group.dropna(subset=[val_col]).copy()
                    if sub_ds.empty: continue
                    best_idx = sub_ds[val_col].idxmax()
                else:
                    # Classification: Select by max val Acc, report test Acc
                    ds_group["score_name"] = "accuracy"
                    val_col = "summary.val_best_rerun/accuracy_mean"
                    test_col = "summary.test_best_rerun/accuracy_mean"
                    test_std = "summary.test_best_rerun/accuracy_std"
                    
                    sub_ds = ds_group.dropna(subset=[val_col]).copy()
                    if sub_ds.empty: continue
                    best_idx = sub_ds[val_col].idxmax()
                
                best_row = sub_ds.loc[best_idx].copy()
                best_row["score"] = best_row[test_col]
                best_row["score_std"] = best_row[test_std]
                best_row["val_score"] = best_row[val_col]
                processed_rows.append(best_row)

    if not processed_rows:
        print("No valid results found for best runs summary.")
        return

    best_runs = pd.DataFrame(processed_rows)
    cols_to_keep = ["dataset_name", "config.model.model_domain", "score_name", "score", "score_std", "val_score", "run_count"] + \
                   [c for c in GROUP_HPARAMS if c in best_runs.columns and c != "config.model.model_domain"]
    
    best_runs = best_runs[cols_to_keep]
    best_runs = best_runs.sort_values(["dataset_name", "config.model.model_domain"])
    best_runs.to_csv(output_file, index=False)
    print(f"Best runs summary (selected by validation) saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Download and aggregate W&B runs.")
    parser.add_argument("--entity", default=WANDB_ENTITY, help="W&B entity name")
    parser.add_argument("--prefix", default=PROJECT_PREFIX, help="Project name prefix")
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--force", action="store_true", help="Force download even if files exist")
    parser.add_argument("--skip-aggregation", action="store_true", help="Skip the aggregation step")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    BEST_RUNS_FILE = args.output_dir / "best_runs_summary.csv"
    
    api = wandb.Api(timeout=120)
    
    print(f"Fetching projects for entity: {args.entity} with prefix: {args.prefix}")
    try:
        all_projects = api.projects(entity=args.entity)
    except Exception as e:
        print(f"Error fetching projects: {e}")
        return

    target_projects = [p.name for p in all_projects if p.name.startswith(args.prefix)]
    
    print(f"Found {len(target_projects)} matching projects.")
    
    for project_name in tqdm(target_projects, desc="Projects"):
        output_file = args.output_dir / f"{project_name}.csv"
        download_project_runs(api, args.entity, project_name, output_file, force=args.force)

    if not args.skip_aggregation:
        print("\nStarting aggregation...")
        aggregate_results(args.output_dir, AGGREGATED_FILE)
        print("\nFinding best runs...")
        save_best_runs(AGGREGATED_FILE, BEST_RUNS_FILE)

if __name__ == "__main__":
    main()
