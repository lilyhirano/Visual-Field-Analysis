# Random Forest Regression for Glaucoma Progression (UW dataset)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Loading the UW visual field data (single CSV)

# per-visit VF tests (includes MS, MS_slope, etc.):

vf_path = "data/UW_VF_Data.csv"

vf_df = pd.read_csv(vf_path)
print("VF data shape:", vf_df.shape)

# Building per-eye baseline table

# Create a unique eye ID to join visits from the same eye:

vf_df["EyeID"] = vf_df["PatientID"].astype(str) + "_" + vf_df["Eye"].astype(str)

# Baseline = smallest visit index or earliest time:

baseline_vf = (
    vf_df
    .sort_values(["EyeID", "Visit"])
    .groupby("EyeID")
    .first()
    .reset_index()
)

print("Baseline VF shape:", baseline_vf.shape)

# For downstream code, use 'merged' as the baseline dataset

merged = baseline_vf.copy()

print("Merged shape:", merged.shape)
merged.head()
