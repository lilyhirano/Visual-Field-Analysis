# severity_model.py


"""
Train a simple CNN model to classify glaucoma severity using PD maps.
Student-written baseline version for the Visual Field project.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --- TensorFlow import with fallback ---
try:
    from tensorflow.keras import layers, models
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "\nTensorFlow is not installed.\n"
        "Please run:\n\n"
        "    pip install tensorflow==2.13\n\n"
    )

# Data PATHS 
DATA_PATH = "data/uw_vf_clean.csv"     
OUT_DIR = "Results"
os.makedirs(OUT_DIR, exist_ok=True)


# DATA LOADING FUNCTION

def load_data():
    """
    Load VF dataset and create severity labels based on mean sensitivity.

    Tries the following columns in order:
    - MS_mean  (if data_cleaning.py computed an average)
    - MS       (baseline mean sensitivity from the original dataset)
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find dataset at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # pick PD_* features 
    pd_cols = [c for c in df.columns if c.startswith("PD_")]
    if len(pd_cols) != 54:
        raise ValueError(f"Expected 54 PD columns, found {len(pd_cols)}.")

    # choose a severity variable 
    if "MS_mean" in df.columns:
        ms = df["MS_mean"]
        print("Using MS_mean as severity variable.")
    elif "MS" in df.columns:
        ms = df["MS"]
        print("MS_mean not found, using MS instead.")
    else:
        raise ValueError(
            "No MS_mean or MS column found in uw_vf_clean.csv.\n"
            "Check data_cleaning.py and make sure at least 'MS' is preserved."
        )

    # SEVERITY DEFINITION (3 CLASSES)
    # mild:   MS > 25 dB
    # moderate: 15–25 dB
    # severe: MS < 15 dB

    
    y = pd.cut(
        ms,
        bins=[-1, 15, 25, 100],
        labels=[2, 1, 0]     # 0 = mild, 1 = moderate, 2 = severe
    ).astype(int)

    # feature matrix from PD values
    X = df[pd_cols].values.astype(float)

    # normalize PD values (simple z-score)
    X = (X - X.mean()) / (X.std() + 1e-6)

    # reshape 54 → (6, 9, 1) for CNN
    X = X.reshape(-1, 6, 9, 1)

    return X, y.values


# CNN MODEL
def build_model():
    """Define a lightweight CNN for 6x9 VF grid classification."""
    model = models.Sequential([
        layers.Input(shape=(6, 9, 1)),
        # use padding="same" so spatial dims don't shrink too much
        layers.Conv2D(16, (3, 3), activation='relu', padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu', padding="same"),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(3, activation='softmax')   # 3 severity classes
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# MAIN TRAINING LOOP
def main():
    print("Loading data...")
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model()
    print("Training CNN...")
    model.fit(X_train, y_train, epochs=8, batch_size=32, verbose=1)

    print("Evaluating...")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {acc:.4f}")

    out_path = os.path.join(OUT_DIR, "severity_model.h5")
    model.save(out_path)
    print("\nSaved CNN model to:", out_path)


if __name__ == "__main__":
    main()
