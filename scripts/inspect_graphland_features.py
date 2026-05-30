# /// script
# requires-python = ">=3.12"
# dependencies = ["pyyaml", "pandas", "numpy", "scikit-learn"]
# ///
"""Inspect GraphLand datasets to compute post-encoding feature dimensions.

Downloads each dataset's info.yaml and features.csv, then computes the
actual num_features after one-hot encoding of categoricals.
"""

import os
import sys
import zipfile
from io import BytesIO
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import OneHotEncoder

ZENODO_RECORD_ID = "16895532"
ZENODO_BASE = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/"

DATASETS = [
    "artnet-exp",
    "artnet-views",
    "avazu-ctr",
    "city-reviews",
    "city-roads-L",
    "city-roads-M",
    "hm-categories",
    "hm-prices",
    "pokec-regions",
    "tolokers-2",
    "twitch-views",
    "web-fraud",
    "web-topics",
    "web-traffic",
]


def download_and_inspect(name: str, timeout: int = 300) -> dict:
    """Download a GraphLand dataset and compute feature dimensions."""
    url = f"{ZENODO_BASE}{name}.zip?"
    print(f"\n{'='*60}")
    print(f"Downloading {name}...")
    req = Request(url, headers={"User-Agent": "graphland-inspect/1.0"})
    with urlopen(req, timeout=timeout) as r:
        blob = r.read()

    print(f"  Downloaded {len(blob) / 1024 / 1024:.1f} MB")

    with zipfile.ZipFile(BytesIO(blob)) as zf:
        # Find info.yaml
        info_data = None
        feats_data = None
        for zi in zf.infolist():
            fname = zi.filename.replace("\\", "/")
            if fname.endswith("info.yaml"):
                info_data = yaml.safe_load(zf.read(zi))
            elif fname.endswith("features.csv"):
                feats_data = pd.read_csv(
                    BytesIO(zf.read(zi)), index_col="node_id"
                )

    if info_data is None or feats_data is None:
        print(f"  ERROR: Missing info.yaml or features.csv")
        return {}

    cat_names = info_data.get("categorical_features_names", [])
    num_names = info_data.get("numerical_features_names", [])
    frac_names = info_data.get("fraction_features_names", [])

    print(f"  Task: {info_data.get('task', 'unknown')}")
    print(f"  Metric: {info_data.get('metric', 'unknown')}")
    print(f"  Nodes: {len(feats_data)}")
    print(f"  Raw features: {len(feats_df.columns) if 'feats_df' in dir() else feats_data.shape[1]}")
    print(f"  Numerical features: {len(num_names)}")
    print(f"  Fraction features: {len(frac_names)}")
    print(f"  Categorical features: {len(cat_names)}")

    # Compute post-encoding dimensions
    num_cols = [c for c in num_names if c in feats_data.columns]
    cat_cols = [c for c in cat_names if c in feats_data.columns]

    numerical_dim = len(num_cols)

    if len(cat_cols) > 0:
        cat_raw = feats_data[cat_cols].values.astype(np.float32)
        encoder = OneHotEncoder(
            drop="if_binary", sparse_output=False, dtype=np.float32
        )
        cat_encoded = encoder.fit_transform(cat_raw)
        categorical_dim = cat_encoded.shape[1]

        print(f"\n  Categorical details:")
        for i, col in enumerate(cat_cols):
            n_unique = len(encoder.categories_[i])
            if n_unique == 2:
                encoded_cols = 1  # drop='if_binary'
            else:
                encoded_cols = n_unique
            print(f"    {col}: {n_unique} unique -> {encoded_cols} encoded cols")
    else:
        categorical_dim = 0

    total_dim = numerical_dim + categorical_dim

    print(f"\n  Post-encoding dimensions:")
    print(f"    Numerical: {numerical_dim}")
    print(f"    Categorical (one-hot): {categorical_dim}")
    print(f"    TOTAL num_features: {total_dim}")

    return {
        "name": name,
        "task": info_data.get("task", "unknown"),
        "metric": info_data.get("metric", "unknown"),
        "nodes": len(feats_data),
        "raw_features": feats_data.shape[1],
        "num_features_post_encoding": total_dim,
        "numerical_dim": numerical_dim,
        "categorical_dim": categorical_dim,
        "num_classes": info_data.get("num_classes", None),
    }


if __name__ == "__main__":
    # Only inspect specific datasets if provided as args, else all
    targets = sys.argv[1:] if len(sys.argv) > 1 else DATASETS

    results = []
    for name in targets:
        try:
            result = download_and_inspect(name)
            if result:
                results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n\n{'='*80}")
    print("SUMMARY: Post-encoding num_features for YAML configs")
    print(f"{'='*80}")
    for r in results:
        print(
            f"  {r['name']:20s}  raw={r['raw_features']:4d}  "
            f"encoded={r['num_features_post_encoding']:4d}  "
            f"(num={r['numerical_dim']}, cat_ohe={r['categorical_dim']})"
        )
