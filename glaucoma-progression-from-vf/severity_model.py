# severity model

"""
severity_model.py

Train a small CNN to predict glaucoma severity from PD maps.

Severity definition (based on MS):
- class 0: MS > -3        (mild / near normal)
- class 1: -12 < MS <= -3 (moderate)
- class 2: MS <= -12      (severe)

Outputs:
- models/severity_cnn.pt
"""

import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

DATA_PATH = "data/uw_vf.csv"
MODEL_PATH = "models/severity_cnn.pt"

GRID_ROWS = 6
GRID_COLS = 9


def values_to_grid(values_1d):
    """Reshape 54-length vector into a (6, 9) grid."""
    values_1d = np.asarray(values_1d)
    assert len(values_1d) == GRID_ROWS * GRID_COLS
    return values_1d.reshape(GRID_ROWS, GRID_COLS)


def ms_to_severity_label(ms: float) -> int:
    """Map MS value to a discrete severity class."""
    if ms > -3:
        return 0  # mild / near normal
    elif ms > -12:
        return 1  # moderate
    else:
        return 2  # severe


class VFDataset(Dataset):
    """PyTorch Dataset for VF PD maps."""

    def __init__(self, df: pd.DataFrame, pd_cols):
        self.df = df.reset_index(drop=True)
        self.pd_cols = pd_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        vals = row[self.pd_cols].values.astype(np.float32)
        grid = values_to_grid(vals)

        # simple z-score normalization
        grid = (grid - grid.mean()) / (grid.std() + 1e-6)

        # add channel dimension: (1, H, W)
        img = np.expand_dims(grid, axis=0)

        label = int(row["severity_label"])

        return torch.tensor(img), torch.tensor(label)


class VFSeverityCNN(nn.Module):
    """Very small CNN for VF severity classification."""

    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # (16, 3, 4)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # (32, 1, 2)
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


def run_epoch(loader, model, criterion, optimizer=None, device="cpu"):
    """One epoch over a dataloader. If optimizer is None, only evaluate."""
    if optimizer is None:
        model.eval()
    else:
        model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        if optimizer is not None:
            optimizer.zero_grad()

        logits = model(X)
        loss = criterion(logits, y)

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += X.size(0)

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples

    return avg_loss, acc


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find dataset at {DATA_PATH}")

    os.makedirs("models", exist_ok=True)

    vf = pd.read_csv(DATA_PATH)

    pd_cols = [c for c in vf.columns if c.startswith("PD_")]
    print("Number of PD columns:", len(pd_cols))

    # create labels
    vf["severity_label"] = vf["MS"].apply(ms_to_severity_label)
    print("Label counts:\n", vf["severity_label"].value_counts())

    dataset = VFDataset(vf, pd_cols)

    # train/validation split
    val_ratio = 0.2
    n_total = len(dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VFSeverityCNN(n_classes=3).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_epochs = 15
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_acc = run_epoch(
            train_loader, model, criterion, optimizer, device=device
        )
        val_loss, val_acc = run_epoch(
            val_loader, model, criterion, optimizer=None, device=device
        )

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_accs.append(tr_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {tr_loss:.3f}, acc {tr_acc:.3f} | "
            f"val loss {val_loss:.3f}, acc {val_acc:.3f}"
        )

    # save model
    torch.save(model.state_dict(), MODEL_PATH)
    print("Saved severity model to:", MODEL_PATH)

    # optional quick plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="train")
    plt.plot(val_accs, label="val")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.title("Accuracy")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
