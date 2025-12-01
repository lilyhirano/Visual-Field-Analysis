# image generator 

import os
import numpy as np
import pandas as pd
from PIL import Image

DATA_PATH = "Data/UW_VF_Data.csv"        
OUT_DIR = "Results/Images"               #

GRID_ROWS = 6
GRID_COLS = 9


def values_to_grid(values):
    """Convert raw PD values into a 6 × 9 grid."""
    arr = pd.to_numeric(values, errors="coerce").fillna(0).astype(float)
    return arr.values.reshape(GRID_ROWS, GRID_COLS)


def normalize_grid(grid):
    """Scale values between 0–255 for grayscale heatmap."""
    g = grid.copy()
    mn, mx = g.min(), g.max()

    if mx - mn < 1e-6:
        return np.zeros_like(g, dtype=np.uint8)

    g = (g - mn) / (mx - mn)
    return (g * 255).astype(np.uint8)


def grid_to_color_image(grid):
    """Convert grayscale grid into RGB heatmap using a simple colormap."""
    # grayscale → color
    h, w = grid.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # simple VF-style colormap: blue → green → yellow → red
    rgb[:, :, 0] = np.clip(grid * 1.5, 0, 255)        # R
    rgb[:, :, 1] = np.clip(255 - grid, 0, 255)        # G
    rgb[:, :, 2] = np.clip(grid * 0.8, 0, 255)        # B

    return Image.fromarray(rgb, mode="RGB").resize((256, 256), Image.NEAREST)


def main(n_examples=200):

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    pd_cols = [c for c in df.columns if c.startswith("PD_")]
    print("PD columns detected:", len(pd_cols))

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Generating {n_examples} fast images…")

    subset = df.head(n_examples)

    for idx, row in subset.iterrows():
        grid = values_to_grid(row[pd_cols])
        norm = normalize_grid(grid)
        img = grid_to_color_image(norm)

        out_path = os.path.join(OUT_DIR, f"vf_{idx}_PD.png")
        img.save(out_path)

    print(f"Done! Saved {n_examples} images → {OUT_DIR}")


if __name__ == "__main__":
    main()
