# Random Forest Regression for Glaucoma Progression (UW dataset)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Loading the UW visual field data:

# per-visit VF tests:

vf_path = "data/uw/VF_Data.csv"      

# per-eye summary including MS_slope:

series_path = "data/uw/Series_Summary.csv"  

vf_df = pd.read_csv(vf_path)
series_df = pd.read_csv(series_path)

print("VF data shape:", vf_df.shape)
print("Series data shape:", series_df.shape)

# Merging visit-level VF data with per-eye summary:
# Creating a unique eye ID to join on:

vf_df["EyeID"] = vf_df["PatientID"].astype(str) + "_" + vf_df["Eye"].astype(str)
series_df["EyeID"] = series_df["PatientID"].astype(str) + "_" + series_df["Eye"].astype(str)

# Baseline = smallest visit index or earliest time:

baseline_vf = (
    vf_df
    .sort_values(["EyeID", "Visit"])
    .groupby("EyeID")
    .first()
    .reset_index()
)

print("Baseline VF shape:", baseline_vf.shape)

# Merge baseline features with summary slope:

merged = baseline_vf.merge(
    series_df[["EyeID", "MS_slope", "Followup_Years"]],
    on="EyeID",
    how="inner"
)

print("Merged shape:", merged.shape)
merged.head()

