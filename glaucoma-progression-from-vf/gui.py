"""
gui.py

Very simple command-line "GUI" to:
- load a VF record
- draw its PD heatmap
- run severity + progression models

This is just a prototype. Later we can move this into a real
Streamlit web app if we want.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import torch
from torch import nn

DATA_PATH = "data/uw_vf.csv"
SEVERITY_MODEL_PATH = "models/severity_cnn.pt"
PROG_MODEL_PATH = "models/ms_slope_rf.pkl"

GRID_ROWS = 6
GRID_COLS = 9


def values_to_grid(values_1d):
    values_1d = np.asarray(values_1d)
    assert len(values_1d) == GRID_ROWS * GRID_COLS
    return values_1d.reshape(GRID_ROWS, GRID_COLS)


class VFSeverityCNN(nn.Module):
    """Same architecture as in severity_model.py."""

    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 1 * 2, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc(x)
        return x


def load_models(device):
    """Load severity CNN and progression RF model."""
    # severity model
    model = VFSeverityCNN(n_classes=3).to(device)
    model.load_state_dict(torch.load(SEVERITY_MODEL_PATH, map_location=device))
    model.eval()

    # progression model
    prog_model = joblib.load(PROG_MODEL_PATH)

    return model, prog_model


def predict_severity(row, pd_cols, model, device):
    """Return (label, probabilities) for one VF record."""
    vals = row[pd_cols].values.astype(np.float32)
    grid = values_to_grid(vals)
    grid = (grid - grid.mean()) / (grid.std() + 1e-6)

    img = np.expand_dims(grid, axis=0)   # (1, H, W)
    img = np.expand_dims(img, axis=0)    # (1, 1, H, W)

    X = torch.tensor(img).to(device)

    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        label = int(logits.argmax(dim=1).item())

    return label, probs


def predict_slope(row, pd_cols, prog_model):
    X = row[pd_cols].values.reshape(1, -1)
    slope = prog_model.predict(X)[0]
    return slope


def show_heatmap(row, pd_cols, idx):
    vals = row[pd_cols].values
    grid = values_to_grid(vals)

    plt.figure(figsize=(4, 3))
    im = plt.imshow(grid, origin="lower", cmap="viridis")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis("off")
    plt.title(f"PD map (row index {idx})")
    plt.show()


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find dataset at {DATA_PATH}")
    if not os.path.exists(SEVERITY_MODEL_PATH):
        raise FileNotFoundError(f"Missing severity model at {SEVERITY_MODEL_PATH}")
    if not os.path.exists(PROG_MODEL_PATH):
        raise FileNotFoundError(f"Missing progression model at {PROG_MODEL_PATH}")

    vf = pd.read_csv(DATA_PATH)
    pd_cols = [c for c in vf.columns if c.startswith("PD_")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    severity_model, prog_model = load_models(device)

    label_names = {0: "mild / near normal", 1: "moderate", 2: "severe"}

    print("Loaded dataset with", len(vf), "rows.")
    print("Enter an integer row index to inspect that VF.")
    print("Type 'q' to quit.\n")

    while True:
        user_input = input("Row index (or 'q'): ").strip()
        if user_input.lower() == "q":
            break

        try:
            idx = int(user_input)
        except ValueError:
            print("Please enter an integer index or 'q'.")
            continue

        if idx < 0 or idx >= len(vf):
            print("Index out of range. Valid range is [0, {}).".format(len(vf)))
            continue

        row = vf.iloc[idx]
        print(
            f"\nPatID {row['PatID']} | Eye {row['Eye']} | "
            f"FieldN {row['FieldN']} | Age {row['Age']:.1f}"
        )
        print(
            f"MS {row['MS']:.2f} | MTD {row['MTD']:.2f} | "
            f"Time_from_Baseline {row['Time_from_Baseline']:.2f}"
        )

        # plot heatmap
        show_heatmap(row, pd_cols, idx)

        # severity prediction
        sev_label, sev_probs = predict_severity(row, pd_cols, severity_model, device)
        print("Predicted severity:", label_names[sev_label])
        print("Class probabilities:", np.round(sev_probs, 3))

        # slope prediction
        slope_pred = predict_slope(row, pd_cols, prog_model)
        print(f"Predicted MS slope (dB/year): {slope_pred:.3f}\n")

    print("Exiting GUI demo.")


if __name__ == "__main__":
    main()
# GUI
