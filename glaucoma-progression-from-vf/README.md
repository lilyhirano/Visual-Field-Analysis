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
