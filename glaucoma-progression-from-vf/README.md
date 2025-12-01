Visual Field Analysis: Glaucoma Progression & Severity Modeling
 
# Overview

This project analyzes visual field (VF) test data from glaucoma patients and builds two predictive models:

A progression model – predicts how fast a patient’s visual field will decline.

A severity model – classifies the current stage of visual field loss from a VF heatmap.

The project also generates visual heatmaps of the visual field, cleans and organizes the dataset, and supports a simple GUI tool for running the models interactively.

# Dataset

The primary dataset is derived from the University of Washington Visual Field (UW VF) database.
Each row represents one VF test and includes:

54 threshold deviation (TD) and pattern deviation (PD) values

Mean Sensitivity (MS)

Follow-up test dates

Global indices like MD and PSD

We focus on PD values because they highlight localized defects used clinically to detect glaucomatous damage.

# Data Cleaning

data_cleaning.py prepares the raw dataset:

Steps:

Load the raw CSV containing all VF tests.

Remove entries with missing MS_slope (the progression ground truth).

Compute MS_mean for severity classification.

Drop unused or fully missing columns.

Save a clean dataset (data/uw_vf_clean.csv) with consistent shape.

Final cleaned dataset size: ≈28,000 rows × 186 columns

# Visual Field Heatmap Generation

image_generation.py converts the 54 PD values into simple 6×9 2D grids and saves them as images.

Why 6×9?

Clinical visual field tests record 54 locations arranged in an irregular pattern—commonly mapped to a 6×9 grid for visualization.

Output:

PNG heatmaps saved in Results/Images/

Useful for:

CNN input

Visual inspection

Presentations and demos

You generated 25–200 images depending on the test runs.

# Progression Model (Random Forest)

progression_model.py predicts the rate of decline of visual field function.

Target variable:

MS_slope (dB/year) → how quickly the eye loses sensitivity.

Features used:

PD values

Global indices

Learnable non-linear interactions

Model:

RandomForestRegressor

Train/test split = 75% / 25%

Output metric: Mean Squared Error (MSE)

Example performance:
Test MSE ≈ 3782.9


This is expected because regression on noisy clinical VF slopes is challenging.
Still, the model captures broad trends in progression.

Output:

A trained model saved locally as:

Results/progression_model.pkl


(Not stored in the GitHub repo because it exceeds 100 MB.)

# Severity Classification Model (CNN)

severity_model.py predicts mild vs moderate vs severe glaucoma using PD heatmaps.

Labels (3-class severity):

Based on Mean Sensitivity (MS_mean):

0 = Mild

1 = Moderate

2 = Severe

Input:

PD values reshaped into (6, 9, 1) image-like tensors

CNN Architecture:

2 convolution layers

MaxPooling

Dense classification head

Softmax output

Training results (your run):
Accuracy ≈ 65.3%


Considering the simplicity of the CNN and noisy clinical data, this is a reasonable baseline.

Output:

A trained model saved locally:

Results/severity_model.h5

# GUI (Interactive Model Demo)

gui.py provides a simple interactive interface to:

Load a patient’s VF record

Display the 6×9 PD heatmap

Run:

Severity prediction via CNN

Progression prediction via Random Forest

Useful for classroom demonstrations or clinician-facing prototypes.
