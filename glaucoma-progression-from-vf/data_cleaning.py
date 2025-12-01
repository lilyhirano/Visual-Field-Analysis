"""
data_cleaning.py

Small helper script to load the UW VF dataset and add a few
basic derived features that we will reuse later.

Right now I only:
- load the CSV
- compute mean TD and mean PD per field
- save a "clean" version

This can be extended later if needed.
"""

import os
import pandas as pd
import numpy as np

RAW_PATH = "data/uw_vf.csv"
CLEAN_PATH = "data/uw_vf_clean.csv"


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a couple of simple summary features for TD and PD."""
    td_cols = [c for c in df.columns if c.startswith("TD_")]
    pd_cols = [c for c in df.columns if c.startswith("PD_")]

    # mean TD and PD across all positions
    df = df.copy()
    df["TD_mean"] = df[td_cols].mean(axis=1)
    df["PD_mean"] = df[pd_cols].mean(axis=1)

    return df


def main():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Could not find dataset at {RAW_PATH}")

    print("Loading dataset...")
    vf = pd.read_csv(RAW_PATH)
    print("Original shape:", vf.shape)

    vf_clean = add_basic_features(vf)

    os.makedirs(os.path.dirname(CLEAN_PATH), exist_ok=True)
    vf_clean.to_csv(CLEAN_PATH, index=False)

    print("Saved cleaned dataset to:", CLEAN_PATH)
    print("Final shape:", vf_clean.shape)


if __name__ == "__main__":
    main()
