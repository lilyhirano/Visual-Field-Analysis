# image generation 

"""
image_generation.py

Convert VF numeric values (Sens / TD / PD) into simple 2D heatmaps.

Right now:
- assumes 54 locations -> reshaped into a 6x9 grid
- saves PD maps as PNG images for the first N rows
"""

import os
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

DATA_PATH = "data/uw_vf.csv"
OUT_DIR = "data/vf_images"

GRID_ROWS = 6
GRID_COLS = 9


def values_to_grid(values_1d):
    """Reshape 54-length vector into a (6, 9) grid."""
    values_1d = np.asarray(values_1d)
    assert len(values_1d) == GRID_ROWS * GRID_COLS
    return values_1d.reshape(GRID_ROWS, GRID_COLS)


def save_vf_image(row, cols, out_path, cmap="viridis"):
    """Save a heatmap for the given row and column set."""
    grid = values_to_grid(row[cols].values)

    plt.figure(figsize=(3, 2.5))
    plt.imshow(grid, origin="lower", cmap=cmap)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def quick_gallery():
    """Show a few saved images as a small sanity check."""
    paths = sorted(glob(os.path.join(OUT_DIR, "vf_*.png")))[:6]
    if not paths:
        print("No images found in", OUT_DIR)
        return

    plt.figure(figsize=(10, 4))
    for j, p in enumerate(paths):
        img = Image.open(p)
        plt.subplot(2, 3, j + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(os.path.basename(p))
    plt.tight_layout()
    plt.show()


def main(n_examples: int = 200):
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find dataset at {DATA_PATH}")

    os.makedirs(OUT_DIR, exist_ok=True)

    vf = pd.read_csv(DATA_PATH)
    pd_cols = [c for c in vf.columns if c.startswith("PD_")]
    print("Number of PD columns:", len(pd_cols))

    subset = vf.head(n_examples)

    for idx, (i, r) in enumerate(subset.iterrows()):
        filename = f"vf_{i}_PD.png"
        path = os.path.join(OUT_DIR, filename)
        save_vf_image(r, pd_cols, path)

    print(f"Saved {len(subset)} PD VF images to {OUT_DIR}")

    # showing a few examples
    quick_gallery()


if __name__ == "__main__":
    main()
