# image_generation.py


import os
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Data paths (keep the old CSV path) 
DATA_PATH = "data/uw_vf.csv"      
OUT_DIR = "Results/Images"        

# grid layout (Humphrey 24-2 has 54 test points) 
GRID_ROWS = 6
GRID_COLS = 9


def values_to_grid(values_1d: np.ndarray) -> np.ndarray:
    """
    Reshape a 1D vector of length 54 into a (6, 9) grid.

    The UW VF file has PD_1 ... PD_54.
    We just map them row-wise into a heatmap.
    """
    values_1d = np.asarray(values_1d, dtype=float)
    assert len(values_1d) == GRID_ROWS * GRID_COLS, "Expected 54 VF test points."
    return values_1d.reshape(GRID_ROWS, GRID_COLS)


def save_vf_image(row: pd.Series, cols: list[str], out_path: str, cmap: str = "viridis") -> None:
    """
    Save a single VF heatmap for a given row of the dataframe.

    Parameters
    ----------
    row : pd.Series
        One row from the dataframe (one visual field exam).
    cols : list[str]
        Column names to pull values from (e.g., all PD_* columns).
    out_path : str
        Where to write the PNG file.
    cmap : str
        Matplotlib colormap name.
    """
    values = row[cols].values
    grid = values_to_grid(values)

    # simple heatmap plot:
    
    plt.figure(figsize=(3, 2.5))
    plt.imshow(grid, origin="lower", cmap=cmap)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def quick_gallery() -> None:
    """
    Show a few saved images as a small sanity check.

    This just helps confirm that the heatmaps look reasonable.
    """
    paths = sorted(glob(os.path.join(OUT_DIR, "*.png")))[:6]
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


def main(n_examples: int = 200) -> None:
    """
    Main entry point.

    Parameters
    ----------
    n_examples : int
        Number of rows (visual fields) to convert into images.
        If the dataset has fewer rows than this value, we just stop at the end.
    """
    # make sure the dataset exists
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find dataset at {DATA_PATH}")

    # create output directory if needed
    os.makedirs(OUT_DIR, exist_ok=True)

    # load the UW VF dataset
    print(f"Loading dataset from: {DATA_PATH}")
    vf = pd.read_csv(DATA_PATH)

    # select all PD_* columns (pattern: PD_1, PD_2, ..., PD_54)
    pd_cols = [c for c in vf.columns if c.startswith("PD_")]
    print("Number of PD columns:", len(pd_cols))

    if len(pd_cols) != GRID_ROWS * GRID_COLS:
        print(
            f"Warning: expected {GRID_ROWS * GRID_COLS} PD columns, "
            f"but found {len(pd_cols)}. Check the CSV header."
        )

    # only take the first n_examples rows for image generation
    subset = vf.head(n_examples)
    total = len(subset)
    print(f"Generating PD heatmaps for the first {total} rows...")

    # iterate over each visual field exam and save an image
    for idx, (row_idx, row) in enumerate(subset.iterrows(), start=1):
        # try to give each file a readable name (PatID + FieldN if present)
        pat_id = row.get("PatID", "unknown")
        field_n = row.get("FieldN", "na")
        filename = f"vf_pat{pat_id}_field{field_n}_row{row_idx}_PD.png"

        out_path = os.path.join(OUT_DIR, filename)
        save_vf_image(row, pd_cols, out_path)

        # print progress every 25 images so it does not look frozen
        if idx % 25 == 0 or idx == total:
            print(f"  Saved {idx}/{total} images")

    print(f"Done. Saved {total} PD VF images to: {OUT_DIR}")

    try:
        quick_gallery()
    except Exception as e:
        print("Could not show gallery:", e)


if __name__ == "__main__":
    main(n_examples=25)
