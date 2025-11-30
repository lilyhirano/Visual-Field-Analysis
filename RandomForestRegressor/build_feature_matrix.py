
# Building feature matrix:

# Features:
#   - Baseline Mean Sensitivity (MS)
#   - Age at baseline
#   - Follow-up duration (years)

# First, check what VF columns exist:

print(merged.columns)

# Adjust this list depending on your actual column names:

basic_features = ["MS", "Age", "Followup_Years"]

# Optionally include all 54 sensitivity points if they're named in a pattern
# Here I assume they are named 'S1'...'S54':

sens_cols = [col for col in merged.columns if col.startswith("S")]

feature_cols = basic_features + sens_cols

# Drop rows with missing values in our features or target:

model_df = merged.dropna(subset=feature_cols + ["MS_slope"]).copy()

X = model_df[feature_cols].values
y = model_df["MS_slope"].values

print("Number of samples after dropping NA:", X.shape[0])
print("Number of features:", X.shape[1])
