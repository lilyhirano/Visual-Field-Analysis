# Predicting Glaucoma Severity and Progression from Humphrey Visual Field (VF) Data
## Contributors: David Houshangi, Lily Hirano, Kirk Ehmsen, Yash Maheshwaran, Christian Fernandez

This project explores whether deep learning and classical machine-learning models can detect visual field damage and predict future progression of glaucoma using the UW Humphrey Visual Field (UWHVF) dataset. The dataset contains thousands of real clinical 24-2 visual field tests collected longitudinally across many patients. Each VF test includes numerical sensitivity values at 54 test points, global indices (MTD, PSD, MS), patient attributes, and follow-up time.

The goal of this project is to transform these VF measurements into 2D images, train models to recognize glaucomatous patterns, and forecast the likely rate of visual field loss over time. In the long term, this pipeline will support a simple GUI where users can upload a VF image and receive both a severity estimate and a predicted progression rate.

# Project Objectives
> 1. Predict current visual field damage from a VF image

Convert the 54 TD/PD/Sensitivity points into a 2D heatmap.

Train models to estimate severity using global indices such as:

Mean Total Deviation (MTD)

Mean Sensitivity (MS)

Pattern Standard Deviation (PSD)

Possible outputs:

Continuous regression (e.g., predicted MTD)

Classification (normal / mild / moderate / severe)

> 2. Predict future glaucoma progression

Using longitudinal VF tests per eye, compute: MTD_slope = 𝑑(𝑀𝑇𝐷) / 𝑑t

This slope (dB/year) serves as the progression label.

Models will then be trained to:

Predict slope from early VFs (baseline or first few tests)

Identify fast vs slow progressors

Learn temporal patterns of deterioration

> 3. Build a prototype GUI

A lightweight tool (Streamlit or PyQt) that:

Accepts a VF image upload

Runs the severity + progression models

Displays:


Current loss estimate

Expected progression trend

Optional predicted future VF heatmap

# Repository Structure

## Notebooks Overview

| Notebook | Description |
|---------|-------------|
| `01_data_exploration.ipynb` | Load the UW VF dataset, inspect variables, analyze MS, MTD, TD, PD distributions. |
| `02_image_generation.ipynb` | Convert Sens / PD / TD values into 2D VF heatmaps using interpolation. |
| `03_severity_model.ipynb` | Train CNN to predict MS severity or classify severity categories. |
| `04_progression_model.ipynb` | Build regression model to estimate MS slope (glaucoma progression). |
| `05_gui_demo.ipynb` | Prototype of GUI for uploading VF and predicting severity + progression. |

glaucoma-progression-from-vf/

data/

notebooks/

data_exploration.ipynb

image_generation.ipynb

severity_model.ipynb

progression_model.ipynb

gui_demo.ipynb

models/

cnn_severity.py

slope_regressor.py

utils.py

visualization/

vf_plotting.py

gui/

app.py   

requirements.txt

README.md

README_DATA.md


# Dataset Overview (UWHVF)

Each row in the dataset represents one VF exam for one eye at one time point. Key fields include:

PatID: patient identifier

Eye: left/right

FieldN: exam number (1st, 2nd, …)

Age: age at exam

Time_from_Baseline: years since first exam

Global indices:

MS, MTD, PSD, cluster measures

54 point-wise measures:

Sens_1 … Sens_54

TD_1 … TD_54

PD_1 … PD_54

The presence of multiple tests per eye makes it ideal for forecasting progression.

# Methods

1. VF Image Construction

Use 54 fixed (x,y) coordinates of HFA 24-2 grid

Interpolate to a smooth 2D heatmap

Normalize intensity values

Create 224×224 grayscale images suitable for CNN models

2. Severity Modeling

Baseline models:

Random Forest Regressor (predict MTD)

Linear/Logistic Regression

XGBoost

Deep models:

Simple CNN for VF images

CNN + dense layers for severity classification/regression

3. Progression Modeling

Compute per-eye MTD_slope using linear regression

Predict slope from:

first VF

early VF sequence

VF images

patient age / PSD / MS

Advanced:

CNN encoder → LSTM for sequence prediction

U-Net-style decoder to generate future VF maps

# Results

Add here

# Future Features

Multi-modal fusion (fundus + VF)

Automated reliability index estimation

Individualized prediction intervals

Uncertainty estimation (MC dropout or deep ensembles)

# Setup
```bash
git clone https://github.com/<your-username>/glaucoma-progression-from-vf
cd glaucoma-progression-from-vf
pip install -r requirements.txt

```

# Citation 

UWHVF: A Real-World, Open Source Dataset of Perimetry Tests From the Humphrey Field Analyzer
