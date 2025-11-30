# Random Forest Regression for Glaucoma Progression (UW dataset)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the UW visual field data (single CSV)

vf_path = "data/UW_VF_Data.csv"

vf_df = pd.read_csv(vf_path)
print("VF data shape:", vf_df.shape)
print("Columns:", vf_df.columns)

# 2. Build per-eye baseline table
# Use PatID (patient id) and Eye (OD/OS) to define a unique eye

vf_df["EyeID"] = vf_df["PatID"].astype(str) + "_" + vf_df["Eye"].astype(str)

# Baseline = earliest time from baseline for each EyeID
baseline_vf = (
    vf_df
    .sort_values(["EyeID", "Time_from_Baseline"])
    .groupby("EyeID")
    .first()
    .reset_index()
)

print("Baseline VF shape:", baseline_vf.shape)

# For downstream code, we will refer to this as 'merged'
merged = baseline_vf.copy()
print("Merged shape:", merged.shape)
merged.head()
