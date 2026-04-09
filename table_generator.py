"""Utilities for fetching W&B runs and aggregating metrics across seeds."""

import ast
import json
import os
import threading
import time

import numpy as np
import pandas as pd
import wandb
from tqdm.auto import tqdm
from wandb.errors import CommError


# ---------------------------------------------------------------------
# Helper to flatten nested config dicts
# ---------------------------------------------------------------------
def flatten_config(config, parent_key="", sep="."):
    """Flatten a nested dictionary by joining keys with a separator.

    Parameters
    ----------
    config : dict
        The nested dictionary to flatten.
    parent_key : str, optional
        Prefix for the flattened keys (used in recursion).
    sep : str, optional
        Separator to join nested keys.

    Returns
    -------
    dict
        A flat dictionary with joined keys.
    """
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# ---------------------------------------------------------------------
# Thread-local storage for wandb API instances (reuse across fetches)
# ---------------------------------------------------------------------
_thread_local = threading.local()


def _get_thread_api(timeout=180):
    """Get or create a thread-local wandb API instance.

    Parameters
    ----------
    timeout : int, optional
        Timeout in seconds for the wandb API client.

    Returns
    -------
    wandb.Api
        A thread-local wandb API instance.
    """
    if not hasattr(_thread_local, "api"):
        _thread_local.api = wandb.Api(timeout=timeout)
    return _thread_local.api


def _is_retryable_error(e):
    """Check if an exception is retryable (rate limit, timeout, server errors).

    Parameters
    ----------
    e : Exception
        The exception to inspect.

    Returns
    -------
    bool
        True if the error is transient and the request should be retried.
    """
    if isinstance(e, CommError):
        return True
    error_str = str(e).lower()
    if hasattr(e, "response") and hasattr(e.response, "status_code"):
        return e.response.status_code in [429, 500, 502, 503, 504]
    retryable_patterns = [
        "429",
        "500",
        "502",
        "503",
        "504",
        "timeout",
        "timed out",
        "server error",
        "try again",
        "connection",
        "reset",
    ]
    return any(p in error_str for p in retryable_patterns)


# ---------------------------------------------------------------------
# Extract data from a run object (already loaded, no additional API call)
# ---------------------------------------------------------------------
def _extract_run_data(run):
    """Extract data from a wandb Run object.

    The run object is already loaded from iteration -- we just extract its
    data. Accessing config/summary may trigger lazy loading but is faster
    than re-fetching.

    Parameters
    ----------
    run : wandb.apis.public.Run
        A wandb run object obtained from API iteration.

    Returns
    -------
    tuple
        ``("success", row_dict)`` on success or ``("failed", (run_id, error_msg))``
        on failure.
    """
    try:
        cfg = run.config.copy() if run.config else {}
        cfg_flat = flatten_config(cfg)

        row = {
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
        }

        # Add all summary metrics (only simple types)
        if run.summary:
            for key, value in run.summary.items():
                if isinstance(value, (int, float, str, bool)) or value is None:
                    row[f"summary.{key}"] = value

        # Add all config parameters
        row.update(cfg_flat)
        return ("success", row)

    except Exception as e:
        return ("failed", (run.id, str(e)))


# ---------------------------------------------------------------------
# Load runs from W&B or from a cached CSV - STREAMING APPROACH
# ---------------------------------------------------------------------
def _fetch_single_project(
    wandb_username,
    wandb_project,
    csv_filename,
    existing_run_ids,
    existing_columns,
    force_load,
    save_csv,
    filters,
    batch_size,
    per_page,
    fetch_recent_only,
    early_stop_threshold,
    project_label=None,
):
    """Fetch runs from a single W&B project and append to the shared CSV.

    Parameters
    ----------
    wandb_username : str
        W&B username / entity.
    wandb_project : str
        W&B project name.
    csv_filename : str
        Path to the output CSV file.
    existing_run_ids : set
        Run IDs already present in the CSV.
    existing_columns : list or None
        Column names of the existing CSV, or None if no CSV exists yet.
    force_load : bool
        If True, ignore existing CSV data.
    save_csv : bool
        If True, persist results to CSV incrementally.
    filters : dict
        Filters passed to the W&B API.
    batch_size : int
        Number of runs to buffer before flushing to CSV.
    per_page : int
        Runs per API page request.
    fetch_recent_only : bool
        If True, fetch newest runs first and early-stop.
    early_stop_threshold : int
        Consecutive existing runs before early stopping.
    project_label : str or None, optional
        Human-readable label for progress messages.

    Returns
    -------
    tuple
        ``(existing_run_ids, existing_columns)`` updated after fetching.
    """
    runs_path = f"{wandb_username}/{wandb_project}"
    max_retries = 10
    retry_delay = 5.0
    label = project_label or wandb_project

    for attempt in range(max_retries):
        # Refresh existing IDs from CSV at the start of every retry
        if os.path.exists(csv_filename) and not force_load:
            try:
                df_existing = pd.read_csv(csv_filename, low_memory=False)
                existing_run_ids = set(df_existing["run_id"].astype(str))
                existing_columns = df_existing.columns.tolist()
            except Exception:
                pass

        if attempt == 0:
            print(f"\n{'=' * 60}")
            print(f"▶ Project: {runs_path}")
            print(
                f"▶ CSV has {len(existing_run_ids)} runs already saved (across all projects)"
            )

        try:
            api = wandb.Api(timeout=300)
            total_runs = len(api.runs(runs_path, filters=filters))
            # We cannot cheaply compute exact new-to-fetch because existing_run_ids
            # is the global set; just report the project total.
            if attempt == 0:
                print(f"▶ Total runs in project: {total_runs}")
        except Exception as e:
            print(f"⚠ Could not get run count: {e}")
            total_runs = None

        order_str = "-created_at" if fetch_recent_only else None
        mode_str = "newest-first" if fetch_recent_only else "default order"
        print(
            f"▶ Attempt {attempt + 1}/{max_retries}: Streaming runs ({mode_str}, batch={batch_size}, page={per_page})..."
        )
        if fetch_recent_only:
            print(
                f"   Early stop after {early_stop_threshold} consecutive existing runs"
            )

        records = []
        failed_runs = []
        iter_count = 0
        new_count = 0
        saved_this_attempt = 0
        consecutive_existing = 0
        early_stopped = False

        pbar = tqdm(desc=f"Fetching [{label}]", unit="runs", total=total_runs)

        try:
            api = wandb.Api(timeout=300)
            runs = api.runs(
                runs_path, filters=filters, per_page=per_page, order=order_str
            )

            for run in runs:
                run_id = str(run.id)
                iter_count += 1

                if run_id in existing_run_ids:
                    consecutive_existing += 1
                    pbar.set_postfix(
                        {
                            "new": new_count,
                            "consec_exist": consecutive_existing,
                            "saved": saved_this_attempt,
                        }
                    )
                    if (
                        fetch_recent_only
                        and consecutive_existing >= early_stop_threshold
                    ):
                        early_stopped = True
                        print(
                            f"\n▶ Early stop: hit {consecutive_existing} consecutive existing runs"
                        )
                        break
                    continue

                consecutive_existing = 0
                pbar.update(1)

                status, result = _extract_run_data(run)

                if status == "success":
                    result["wandb_project"] = wandb_project
                    records.append(result)
                    existing_run_ids.add(run_id)
                    new_count += 1
                else:
                    failed_runs.append(result)

                pbar.set_postfix(
                    {
                        "new": new_count,
                        "consec_exist": consecutive_existing,
                        "saved": saved_this_attempt,
                    }
                )

                if save_csv and len(records) >= batch_size:
                    existing_columns = _save_batch_to_csv(
                        records, csv_filename, existing_columns
                    )
                    saved_this_attempt += len(records)
                    records = []

            pbar.close()

            if save_csv and records:
                existing_columns = _save_batch_to_csv(
                    records, csv_filename, existing_columns
                )
                saved_this_attempt += len(records)

            if early_stopped:
                print(f"▶ [{label}] Complete (early stopped)!")
            else:
                print(f"\n▶ [{label}] Complete!")
            print(f"   Iterated: {iter_count} runs")
            print(f"   New runs saved this session: {saved_this_attempt}")
            print(f"   Total in CSV: {len(existing_run_ids)}")
            if failed_runs:
                print(f"   Failed: {len(failed_runs)}")

            return existing_run_ids, existing_columns

        except Exception as e:
            pbar.close()

            if save_csv and records:
                existing_columns = _save_batch_to_csv(
                    records, csv_filename, existing_columns
                )
                saved_this_attempt += len(records)
                records = []

            if _is_retryable_error(e) and attempt < max_retries - 1:
                wait_time = (
                    retry_delay * (1.5**attempt) + np.random.random() * 3
                )
                print(
                    f"\n⚠ [{label}] Error at iteration {iter_count}: {str(e)[:80]}"
                )
                print(f"   Saved {saved_this_attempt} runs this attempt")
                print(
                    f"   Retrying in {wait_time:.1f}s... (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(wait_time)
            else:
                print(
                    f"\n✗ [{label}] Fatal error after {attempt + 1} attempts: {e}"
                )
                break

    return existing_run_ids, existing_columns


def load_results_dataframe(
    wandb_username,
    wandb_project,
    csv_filename="wandb_results.csv",
    force_load=False,
    save_csv=True,
    filters=None,
    batch_size=500,
    per_page=50,
    fetch_recent_only=True,
    early_stop_threshold=100,
):
    """Load results from one or more W&B projects into a single DataFrame/CSV.

    Parameters
    ----------
    wandb_username : str
        W&B username / entity.
    wandb_project : str or list of str
        W&B project name(s).
    csv_filename : str, optional
        Output CSV filename.
    force_load : bool, optional
        If True, reload all runs even if CSV exists.
    save_csv : bool, optional
        If True, save results to CSV incrementally.
    filters : dict or None, optional
        Filters passed to the W&B API (applied to every project).
    batch_size : int, optional
        Runs to process before saving to CSV.
    per_page : int, optional
        Runs per API page request (lower = more stable).
    fetch_recent_only : bool, optional
        If True, fetch newest runs first and stop early when hitting
        existing runs. Much faster for incremental updates.
    early_stop_threshold : int, optional
        When *fetch_recent_only* is True, stop after hitting this many
        consecutive runs that are already in CSV.

    Returns
    -------
    pandas.DataFrame
        Combined DataFrame of all fetched runs.
    """
    if filters is None:
        filters = {}

    # Normalise to a list so the rest of the logic is uniform
    projects = (
        [wandb_project]
        if isinstance(wandb_project, str)
        else list(wandb_project)
    )

    # Handle force_load backup once, before any project is fetched
    if force_load and os.path.exists(csv_filename):
        backup_name = csv_filename.replace(".csv", "_backup.csv")
        os.rename(csv_filename, backup_name)
        print(f"▶ Force reload: backed up existing CSV to {backup_name}")

    # Seed the existing-run set from the CSV (shared across all projects)
    existing_run_ids: set = set()
    existing_columns = None

    if os.path.exists(csv_filename) and not force_load:
        try:
            df_existing = pd.read_csv(csv_filename, low_memory=False)
            existing_run_ids = set(df_existing["run_id"].astype(str))
            existing_columns = df_existing.columns.tolist()
            print(f"▶ CSV has {len(existing_run_ids)} runs already saved")
        except Exception as e:
            print(f"⚠ Could not read existing CSV: {e}")

    print(f"▶ Will fetch from {len(projects)} project(s): {projects}")

    for proj in projects:
        existing_run_ids, existing_columns = _fetch_single_project(
            wandb_username=wandb_username,
            wandb_project=proj,
            csv_filename=csv_filename,
            existing_run_ids=existing_run_ids,
            existing_columns=existing_columns,
            force_load=False,  # backup already handled above
            save_csv=save_csv,
            filters=filters,
            batch_size=batch_size,
            per_page=per_page,
            fetch_recent_only=fetch_recent_only,
            early_stop_threshold=early_stop_threshold,
        )

    # Return the combined CSV
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename, low_memory=False)
        print(f"\n▶ Final combined DataFrame shape: {df.shape}")
        return df
    return pd.DataFrame()


def _save_batch_to_csv(records, csv_filename, existing_columns=None):
    """Save a batch of records to CSV, handling column alignment.

    Parameters
    ----------
    records : list of dict
        Rows to append.
    csv_filename : str
        Path to the CSV file.
    existing_columns : list of str or None, optional
        Known column order of the existing CSV.

    Returns
    -------
    list of str
        Updated column list after the write.
    """
    if not records:
        return existing_columns

    df_new = pd.DataFrame(records)

    if os.path.exists(csv_filename):
        # Read existing to align columns
        if existing_columns is None:
            df_existing = pd.read_csv(csv_filename, nrows=0)
            existing_columns = df_existing.columns.tolist()

        # Find new columns
        new_cols = [c for c in df_new.columns if c not in existing_columns]

        if new_cols:
            # Add new columns to existing file
            df_existing = pd.read_csv(csv_filename, low_memory=False)
            for col in new_cols:
                df_existing[col] = None
            df_existing.to_csv(csv_filename, index=False)
            existing_columns = df_existing.columns.tolist()

        # Align new data columns and append
        for col in existing_columns:
            if col not in df_new.columns:
                df_new[col] = None
        df_new = df_new[existing_columns]
        df_new.to_csv(csv_filename, mode="a", header=False, index=False)
    else:
        # Create new file
        df_new.to_csv(csv_filename, index=False)
        existing_columns = df_new.columns.tolist()

    return existing_columns


# ---------------------------------------------------------------------
# Helper: serialize values into a *string* key for grouping
# ---------------------------------------------------------------------
def _serialize_for_grouping(val):
    """Convert a Python object into a stable string for pandas grouping.

    Parameters
    ----------
    val : object
        Any value (scalar, list, dict, ndarray, etc.).

    Returns
    -------
    str
        A deterministic string representation suitable for ``groupby``.
    """
    # Preserve missingness
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "__NaN__"

    # Numpy arrays -> list
    if isinstance(val, np.ndarray):
        return "__ndarray__:" + json.dumps(val.tolist(), sort_keys=False)

    # Dicts -> sorted by key for stability
    if isinstance(val, dict):
        return "__dict__:" + json.dumps(val, sort_keys=True, default=str)

    # Lists/tuples/sets -> serialized list
    if isinstance(val, (list, tuple, set)):
        seq = sorted(list(val)) if isinstance(val, set) else list(val)
        return "__seq__:" + json.dumps(seq, sort_keys=False, default=str)

    # Fallback: stringify scalars/other objects
    return f"__val__:{repr(val)}"


# ---------------------------------------------------------------------
# Main aggregator
# ---------------------------------------------------------------------
def aggregate_across_seeds(
    df,
    seed_col="dataset.split_params.data_seed",
    metric_prefix="summary.",
    output_filename=None,
    expected_seeds=None,
):
    """Aggregate W&B runs across seeds, computing mean/std/count per metric.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame of individual runs (one row per run).
    seed_col : str, optional
        The seed column to aggregate over.
    metric_prefix : str, optional
        Column-name prefix that identifies metric columns.
    output_filename : str or None, optional
        If given, save the aggregated DataFrame to this CSV path.
    expected_seeds : int or None, optional
        If given, keep only groups with exactly this many seeds.
        If None, inferred from the number of unique values of *seed_col*.

    Returns
    -------
    pandas.DataFrame
        Aggregated DataFrame with mean, std, and count columns per metric.
    """

    all_seed_cols = {seed_col}
    for c in df.columns:
        if "seed" in c.lower():
            all_seed_cols.add(c)

    print(f"▶ Seed column: {seed_col}")
    print(
        f"▶ All seed columns excluded from grouping: {sorted(all_seed_cols)}"
    )

    # Backfill missing seeds from run_name (e.g. "..._seed9")
    # Also derive a config key for those runs, since their wandb config
    # columns are typically empty and they'd otherwise collapse into one
    # giant group per project.
    _RUN_CONFIG_KEY = "_run_config_key"
    df[_RUN_CONFIG_KEY] = ""

    missing_mask = df[seed_col].isna()
    n_missing = missing_mask.sum()
    if n_missing > 0 and "run_name" in df.columns:
        extracted = df.loc[missing_mask, "run_name"].str.extract(
            r"_seed(\d+)", expand=False
        )
        recovered = extracted.notna().sum()
        df.loc[missing_mask, seed_col] = pd.to_numeric(
            extracted, errors="coerce"
        )
        backfilled_mask = missing_mask & df[seed_col].notna()
        df.loc[backfilled_mask, _RUN_CONFIG_KEY] = df.loc[
            backfilled_mask, "run_name"
        ].str.replace(r"_seed\d+$", "", regex=True)
        still_missing = df[seed_col].isna().sum()
        print(
            f"▶ Backfilled {recovered}/{n_missing} missing seeds from run_name"
        )
        if still_missing > 0:
            print(f"▶ Dropping {still_missing} runs with no seed info at all")
            df = df.dropna(subset=[seed_col])

    if expected_seeds is None:
        expected_seeds = int(df[seed_col].nunique(dropna=True))
    print(f"▶ Expected seeds per group: {expected_seeds}")

    # 3) Identify metric columns -----------------------------------------------------
    metric_cols = [c for c in df.columns if c.startswith(metric_prefix)]
    if not metric_cols:
        raise ValueError(
            f"No metric columns found with prefix '{metric_prefix}'"
        )
    print(
        f"▶ Found {len(metric_cols)} metric columns with prefix '{metric_prefix}'"
    )

    # 4) Determine initial candidate grouping columns --------------------------------
    exclude_cols = set(all_seed_cols) | {"run_id", "run_name", "state"}
    exclude_cols.update(metric_cols)

    candidate_grouping = [c for c in df.columns if c not in exclude_cols]

    # Drop columns that are per-run artifacts, not experiment config
    bad_group_cols = {
        "dataset.baselines.svm.pipeline",
        "dataset.baselines.elastic_net.pipeline",
        "trainer.devices",
        "transforms.hopse_encoding.cuda",
    }
    # Also drop AvgTime columns (runtime metrics, not config)
    for c in list(candidate_grouping):
        if c.startswith(("AvgTime/", "model/params/")):
            bad_group_cols.add(c)
    candidate_grouping = [
        c for c in candidate_grouping if c not in bad_group_cols
    ]

    print(
        f"▶ Initial grouping candidates (after dropping bad cols): {len(candidate_grouping)}"
    )

    # 5) (Optional) drop columns that are unique per run -----------------------------
    # This keeps things like logger.wandb.id, ckpt_path, etc. out of grouping.
    # Serialize candidate columns first to handle unhashable types (lists, dicts, etc.)
    df_temp = df.copy()
    for col in candidate_grouping:
        df_temp[col] = df_temp[col].apply(_serialize_for_grouping)

    n_rows = len(df)
    max_expected_groups = n_rows // expected_seeds
    grouping_cols = []
    dropped_nunique = []
    for col in candidate_grouping:
        nunique = df_temp[col].nunique(dropna=False)
        if nunique <= max_expected_groups:
            grouping_cols.append(col)
        else:
            dropped_nunique.append((col, nunique))

    if dropped_nunique:
        dropped_nunique.sort(key=lambda x: -x[1])
        print(
            f"▶ Dropped {len(dropped_nunique)} per-run columns (nunique > {max_expected_groups}):"
        )
        for col, nu in dropped_nunique[:10]:
            print(f"    nunique={nu:6d}  {col}")
        if len(dropped_nunique) > 10:
            print(f"    ... and {len(dropped_nunique) - 10} more")
    print(
        f"▶ Final grouping columns (after nunique filter): {len(grouping_cols)}"
    )

    # 6) Make a copy, serialize grouping cols, and coerce metrics to numeric --------
    df_group = df.copy()

    # Serialize grouping columns so lists/dicts/arrays are safe keys
    for col in grouping_cols:
        df_group[col] = df_group[col].apply(_serialize_for_grouping)

    # Coerce metrics to numeric; non-numeric become NaN
    for col in metric_cols:
        df_group[col] = pd.to_numeric(df_group[col], errors="coerce")

    # Drop metric columns that are entirely NaN after coercion
    numeric_metric_cols = [c for c in metric_cols if df_group[c].notna().any()]
    dropped = sorted(set(metric_cols) - set(numeric_metric_cols))
    print(f"▶ Numeric metric columns kept: {len(numeric_metric_cols)}")
    if dropped:
        print(
            f"▶ Dropped {len(dropped)} all-NaN / non-numeric metric columns (e.g.): {dropped[:5]}"
        )

    if not numeric_metric_cols:
        raise ValueError(
            "After filtering, no numeric metric columns remain to aggregate."
        )

    # 7) Build aggregation dict and group -------------------------------------------
    agg_dict = {col: ["mean", "std", "count"] for col in numeric_metric_cols}

    grouped = df_group.groupby(grouping_cols, dropna=False, sort=False)

    # 7a) Enforce exactly expected_seeds per group -----------------------------------
    group_sizes = grouped.size()
    matching_groups = group_sizes[group_sizes == expected_seeds].index

    if len(matching_groups) == 0:
        size_dist = group_sizes.value_counts().sort_index()
        raise ValueError(
            f"No groups found with exactly {expected_seeds} seeds. "
            f"Group-size distribution:\n{size_dist}\nCannot aggregate."
        )

    if len(matching_groups) < len(group_sizes):
        n_filtered = len(group_sizes) - len(matching_groups)
        print(
            f"▶ Filtering: {n_filtered} groups removed (did not have exactly {expected_seeds} seeds)"
        )
        print(
            f"▶ Keeping: {len(matching_groups)} groups with exactly {expected_seeds} seeds"
        )

        df_group_filtered = (
            df_group.set_index(grouping_cols)
            .loc[matching_groups]
            .reset_index()
        )
        grouped = df_group_filtered.groupby(
            grouping_cols, dropna=False, sort=False
        )
    else:
        print(
            f"▶ All {len(group_sizes)} groups have exactly {expected_seeds} seeds"
        )

    aggregated = grouped[numeric_metric_cols].agg(agg_dict)

    # 8) Flatten MultiIndex columns --------------------------------------------------
    new_cols = []
    for metric, stat in aggregated.columns:
        if stat == "mean":
            new_cols.append(metric)
        else:
            new_cols.append(f"{metric}_{stat}")

    aggregated.columns = new_cols
    aggregated = aggregated.reset_index()

    print(f"▶ Aggregated shape: {aggregated.shape}")
    print(f"▶ Number of unique experiment configurations: {len(aggregated)}")

    if output_filename:
        aggregated.to_csv(output_filename, index=False)
        print(f"▶ Saved aggregated results to: {output_filename}")

    return aggregated


def diagnose_grouping_conflicts(
    df,
    metric_prefix="summary.",
    seed_cols=None,
):
    """Diagnose columns that vary within core configs and break seed grouping.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame of individual runs.
    metric_prefix : str, optional
        Column-name prefix that identifies metric columns.
    seed_cols : list of str or None, optional
        Seed columns to exclude from grouping. Auto-detected when None.
    """

    # 1) Detect seed columns ---------------------------------------------------------
    if seed_cols is None:
        seed_cols = []
        possible_seed_cols = [
            "seed",
            "dataset.split_params.data_seed",
            "dataset.loader.parameters.data_seed",
        ]
        for col in possible_seed_cols:
            if col in df.columns:
                seed_cols.append(col)
        if not seed_cols:
            seed_cols = [c for c in df.columns if "seed" in c.lower()]

    print(f"▶ Using seed columns (for diagnostics only): {seed_cols}")

    # 2) Metric columns --------------------------------------------------------------
    metric_cols = [c for c in df.columns if c.startswith(metric_prefix)]
    exclude_cols = (
        set(metric_cols) | set(seed_cols) | {"run_id", "run_name", "state"}
    )

    # 3) Candidate non-metric config-ish columns ------------------------------------
    candidate_cols = [c for c in df.columns if c not in exclude_cols]

    # 4) Serialize candidate columns so groupby / nunique won't choke ----------------
    df_ser = df.copy()
    for col in candidate_cols:
        df_ser[col] = df_ser[col].apply(_serialize_for_grouping)

    # 5) Define a "core config" using sensible prefixes ------------------------------
    base_prefixes = [
        "model.backbone.",
        "model.readout.",
        "model.feature_encoder.",
        "model.model_name",
        "model.model_domain",
        "dataset.loader.parameters.",
        "dataset.parameters.",
        "optimizer.parameters.",
    ]
    base_group_cols = []
    base_group_cols += [
        col
        for col in candidate_cols
        if any(col.startswith(p) for p in base_prefixes)
    ]

    # Add a couple of standalone columns if present
    base_group_cols += [
        col for col in ["task_name", "evaluator.task"] if col in candidate_cols
    ]

    # Deduplicate while preserving order
    seen = set()
    base_group_cols = [
        c for c in base_group_cols if not (c in seen or seen.add(c))
    ]

    if not base_group_cols:
        print(
            "⚠ No base_group_cols found with the chosen prefixes. "
            "You may need to adjust base_prefixes."
        )
        return

    print(f"▶ Core config grouping columns ({len(base_group_cols)}):")
    for c in base_group_cols:
        print("   ", c)

    # 6) Group by core config and look at group sizes --------------------------------
    grouped_core = df_ser.groupby(base_group_cols, dropna=False)
    group_sizes = grouped_core.size()

    print(f"\n▶ Number of unique core configs: {len(group_sizes)}")
    print(
        f"▶ Runs per core config: min={group_sizes.min()}, "
        f"max={group_sizes.max()}, mean={group_sizes.mean():.2f}"
    )
    multi_core = group_sizes[group_sizes > 1]

    if multi_core.empty:
        print(
            "⚠ No core config has more than one run. "
            "Either you truly have no repeated configs, or the core "
            "grouping is too fine; try removing some base_prefixes."
        )
        return

    print(
        f"▶ Core configs with >1 run (where seeds *should* aggregate): {len(multi_core)}"
    )

    # Restrict to those multi-run core configs
    idx_multi = multi_core.index
    df_multi = df_ser.set_index(base_group_cols).loc[idx_multi].reset_index()

    # 7) For all other candidate columns, see if they vary *within* a core config ----
    other_cols = sorted(set(candidate_cols) - set(base_group_cols))

    varying_info = []  # (col, frac_groups_vary, max_nunique)

    grouped_multi = df_multi.groupby(base_group_cols, dropna=False)
    n_groups = len(multi_core)

    for col in other_cols:
        nunique_per_group = grouped_multi[col].nunique(dropna=False)
        max_nunique = nunique_per_group.max()
        if max_nunique > 1:
            frac_vary = (nunique_per_group > 1).sum() / n_groups
            varying_info.append((col, frac_vary, int(max_nunique)))

    if not varying_info:
        print("\n▶ No additional columns vary within core configs.")
        print(
            "  That means the issue is likely that the core config itself "
            "is too detailed for seeds to line up."
        )
        return

    varying_info.sort(key=lambda x: x[1], reverse=True)

    print(
        "\n▶ Columns that vary within core configs (and would break seed aggregation):"
    )
    print(
        "   (col, fraction_of_core_configs_where_it_varies, max_nunique_within_a_config)"
    )
    for col, frac, max_n in varying_info[:50]:  # show top 50
        print(f"   {col:60s}  frac={frac:6.3f}, max_nunique={max_n}")

    # Optionally, also print some "safe" columns that never vary
    stable_cols = [
        col for col in other_cols if col not in {v[0] for v in varying_info}
    ]
    if stable_cols:
        print(
            "\n▶ Example columns that are stable within core configs (safe to include if desired):"
        )
        for c in stable_cols[:30]:
            print("   ", c)


# ---------------------------------------------------------------------
# Helper: decode serialized values
# ---------------------------------------------------------------------
def _decode_val(x):
    """Decode a value previously serialized by ``_serialize_for_grouping``.

    Parameters
    ----------
    x : object
        A raw or serialized value.

    Returns
    -------
    object
        The decoded Python value.
    """
    if isinstance(x, str):
        if x == "__NaN__":
            return np.nan
        if x.startswith("__val__:"):
            inner = x[len("__val__:") :]
            # Strip simple quotes if present
            if inner.startswith("'") and inner.endswith("'"):
                return inner[1:-1]
            # Try to literal-eval numbers / booleans
            try:
                return ast.literal_eval(inner)
            except Exception:
                return inner
    return x


# ---------------------------------------------------------------------
# Find best runs by val/f1_macro and create tables per dataset
# ---------------------------------------------------------------------
def get_best_runs_by_val_f1_macro(
    aggregated_df,
    metric_col="summary.val/f1_macro",
    data_name_col="dataset.loader.parameters.data_name",
    method_col="dataset.loader.parameters.method",
    node_sample_ratio_col="dataset.loader.parameters.node_sample_ratio",
    datasets=None,
):
    """Select the best run per dataset/method/ratio and build pivot tables.

    Parameters
    ----------
    aggregated_df : pandas.DataFrame
        Aggregated DataFrame (output of :func:`aggregate_across_seeds`).
    metric_col : str, optional
        Column used to rank runs (higher is better).
    data_name_col : str, optional
        Column containing dataset names.
    method_col : str, optional
        Column containing method names.
    node_sample_ratio_col : str, optional
        Column containing node-sample ratios.
    datasets : list of str or None, optional
        Restrict to these datasets. If None, use all available.

    Returns
    -------
    dict
        Mapping from dataset name to a pivot table of ``"mean +/- std"`` strings.
    """
    # Make a copy to avoid modifying the original
    df = aggregated_df.copy()

    # Check required columns exist
    required_cols = [
        metric_col,
        data_name_col,
        method_col,
        node_sample_ratio_col,
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Decode serialized values for the grouping columns
    for col in [data_name_col, method_col, node_sample_ratio_col]:
        if col in df.columns:
            df[col] = df[col].apply(_decode_val)

    # Ensure metric is numeric
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")

    # Filter out rows with NaN in required columns
    df = df.dropna(
        subset=[data_name_col, method_col, node_sample_ratio_col, metric_col]
    )

    if len(df) == 0:
        raise ValueError("No valid rows found after filtering")

    # Get datasets to process
    if datasets is None:
        datasets = sorted(df[data_name_col].dropna().unique())
    else:
        # Filter to only requested datasets
        df = df[df[data_name_col].isin(datasets)]
        if len(df) == 0:
            raise ValueError(f"No data found for datasets: {datasets}")

    # Find the std column for the metric
    metric_std_col = f"{metric_col}_std"
    df[metric_std_col] = pd.to_numeric(df[metric_std_col], errors="coerce")

    # Group by dataset, method, and node_sample_ratio, and find the best run
    # (highest val/f1_macro) for each combination
    grouped = df.groupby(
        [data_name_col, method_col, node_sample_ratio_col], dropna=False
    )

    # For each group, get the row with the maximum metric value
    best_runs = []
    for (_data_name, _method, _ratio), group in grouped:
        # Find the row with the maximum metric value
        best_idx = group[metric_col].idxmax()
        best_run = group.loc[best_idx].copy()
        best_runs.append(best_run)

    best_df = pd.DataFrame(best_runs)

    # Create pivot tables for each dataset (combined mean ± std)
    result_tables = {}

    for dataset in datasets:
        dataset_data = best_df[best_df[data_name_col] == dataset].copy()

        if len(dataset_data) == 0:
            print(f"⚠ No data found for dataset: {dataset}")
            continue

        # Create pivot tables for mean and std
        pivot_table_mean = dataset_data.pivot_table(
            values=metric_col,
            index=method_col,
            columns=node_sample_ratio_col,
            aggfunc="first",
        )

        pivot_table_std = dataset_data.pivot_table(
            values=metric_std_col,
            index=method_col,
            columns=node_sample_ratio_col,
            aggfunc="first",
        )

        # Sort methods and ratios for better readability
        if len(pivot_table_mean) > 0:
            pivot_table_mean = pivot_table_mean.sort_index()
            pivot_table_mean = pivot_table_mean.sort_index(axis=1)
            pivot_table_std = pivot_table_std.sort_index()
            pivot_table_std = pivot_table_std.sort_index(axis=1)

        # Combine mean and std into formatted strings: "mean ± std"
        combined_table = pivot_table_mean.copy()
        for idx in pivot_table_mean.index:
            for col in pivot_table_mean.columns:
                mean_val = pivot_table_mean.loc[idx, col]
                std_val = pivot_table_std.loc[idx, col]

                if pd.notna(mean_val) and pd.notna(std_val):
                    combined_table.loc[idx, col] = (
                        f"{mean_val:.4f} ± {std_val:.4f}"
                    )
                elif pd.notna(mean_val):
                    combined_table.loc[idx, col] = f"{mean_val:.4f}"
                else:
                    combined_table.loc[idx, col] = "NaN"

        result_tables[dataset] = combined_table

    return result_tables
