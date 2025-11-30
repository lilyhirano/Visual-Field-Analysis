# Random Forest Regression / Severity Prediction for UW Visual Field Data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    this_file = Path(__file__).resolve()
    project_root = this_file.parents[1]
    data_dir = project_root / "data"
    rfr_dir = project_root / "RandomForestRegressor"


    # 1. Load UW visual field data

    vf_path = data_dir / "UW_VF_Data.csv"

    vf_df = pd.read_csv(vf_path)
    print("VF data shape:", vf_df.shape)
    print("Columns:", vf_df.columns)


    # 2. Build per-eye baseline table
    # Use PatID + Eye as unique eye identifier
    # Baseline = earliest Time_from_Baseline per eye

    
    vf_df["EyeID"] = vf_df["PatID"].astype(str) + "_" + vf_df["Eye"].astype(str)

    baseline_vf = (
        vf_df
        .sort_values(["EyeID", "Time_from_Baseline"])
        .groupby("EyeID")
        .first()
        .reset_index()
    )

    print("Baseline VF shape:", baseline_vf.shape)    
    merged = baseline_vf.copy()
    print("Merged shape:", merged.shape)

    # 3. Feature selection
    # Basic features
    
    basic_features = ["MS", "Age", "Time_from_Baseline"]

    # Pattern deviation features (PD_45 ... PD_54, etc.)
    
    pd_features = [c for c in merged.columns if c.startswith("PD_")]

    # (Mean Sensitivity) MS cluster features
    
    cluster_features = [c for c in merged.columns if c.startswith("MS_Cluster")]

    feature_cols = basic_features + pd_features + cluster_features

    print("Using", len(feature_cols), "features.")

    # 4. Build model dataframe (only require target to be present)
    # Use MS (mean sensitivity) as regression target
    
    target_col = "MS"

    # Keep rows where target is not NaN:
    
    mask = merged[target_col].notna()
    y = merged.loc[mask, target_col].values

    # Take features for those rows and fill missing values:
    
    X_df = merged.loc[mask, feature_cols].copy()

    # Fill NaNs in features with column means:
    
    X_df = X_df.fillna(X_df.mean(numeric_only=True))

    X = X_df.values

    print("Model dataframe shape:", X_df.shape)

    # 5. Train/test split:
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 6. Train Random Forest:
    
    rf = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # 7. Evaluation:
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")

    # Save results into the RandomForestRegressor folder:
    
    results_path = rfr_dir / "RFR_results.txt"
    with results_path.open("w") as f:
        f.write("RandomForestRegressor – UW dataset\n")
        f.write("----------------------------------\n")
        f.write(f"MAE:  {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R^2:  {r2:.4f}\n")

    print(f"Saved metrics to {results_path}")

    # 8. Feature importance bar plot
    
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1][:15]  

    plt.figure(figsize=(8, 6))
    plt.bar(range(len(idx)), importances[idx])
    plt.xticks(
        range(len(idx)),
        [feature_cols[i] for i in idx],
        rotation=45,
        ha="right",
    )
    plt.ylabel("Feature Importance")
    plt.title("Random Forest Feature Importances (UW VF baseline)")
    plt.tight_layout()

    fig_path = rfr_dir / "RFR_feature_importance.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"Saved feature importance plot to {fig_path}")

    # 9. PD-feature importance heatmap (top PD features only)
    # Map importances to feature names
    
    fi_series = pd.Series(importances, index=feature_cols)

    # Keep only PD_* features
    pd_importances = fi_series[pd_features]

    if len(pd_importances) > 0:
        # Sort PD features by importance (largest first)
        pd_importances = pd_importances.sort_values(ascending=False)

        # Keep only the top K PD points for readability
        top_k = min(20, len(pd_importances))  # show up to 20
        pd_importances = pd_importances.iloc[:top_k]

        # Normalize to [0, 1] so the color scale is easier to see
        vals = pd_importances.values
        if vals.max() > 0:
            vals = vals / vals.max()

        # Make a wider, short heatmap so labels are readable
        fig_width = 0.6 * top_k + 2  # scale width with number of features
        fig, ax = plt.subplots(figsize=(fig_width, 2.5))

        im = ax.imshow(vals.reshape(1, -1), aspect="auto")

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Relative importance")

        ax.set_xticks(np.arange(top_k))
        ax.set_xticklabels(pd_importances.index, rotation=45, ha="right", fontsize=7)
        ax.set_yticks([0])
        ax.set_yticklabels(["PD importance"], fontsize=8)

        ax.set_title("Random Forest PD Feature Importance (top PD points)")
        fig.tight_layout()

        heat_path = rfr_dir / "RFR_PD_importance_heatmap.png"
        fig.savefig(heat_path, dpi=300)
        plt.close(fig)

        print(f"Saved PD importance heatmap to {heat_path}")
    else:
        print("No PD_* features found – skipping heatmap.")


if __name__ == "__main__":
    main()
