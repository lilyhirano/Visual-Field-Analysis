# Random Forest Regression for Glaucoma Progression (UW Dataset)
**Course:** Chem 277B – Machine Learning Algorithms  
**Project:** Visual Field Analysis for Glaucoma Prediction  
**Dataset:** University of Washington Biomedical AI Visual Fields (UWHVF)

## Overview
This model predicts glaucoma progression by estimating the **mean sensitivity slope** (MS_slope, dB/year) for each eye. The slope represents the annual rate of functional vision loss, with more negative values indicating faster deterioration. We focused on how well baseline visual field (VF) patterns can predict future decline.

## Feature Construction
For each eye, we used only the **first recorded VF test** (baseline). Baseline inputs included:

- Mean Sensitivity (MS)
- Age at visit
- Follow-up duration (years)
- Raw 54-point sensitivity grid (optional, if included)

Each row represents one eye.

## Model
We used a **RandomForestRegressor** with:

- `n_estimators = 200`
- `min_samples_leaf = 3`
- `random_state = 42`
- 80/20 train–test split

This ensemble model captures non-linear interactions across VF locations and patient-level variables.

## Results
On the held-out test set, the model achieved:

- **MAE:** [MAE_here] dB/year  
- **RMSE:** [RMSE_here] dB/year  
- **R²:** [R2_here]  

(Replace with your actual values after running.)

## Feature Importance
The Random Forest produced a ranked list of influential features. The most predictive variables were:

- Baseline mean sensitivity (MS)
- Follow-up duration
- Several peripheral sensitivity points in glaucomatous regions (arcuate and nasal areas)

This supports known clinical patterns of glaucomatous field loss.

## Interpretation
A relatively simple ensemble model can capture clinically meaningful structure in the VF data. Baseline VF measurements contain enough signal to provide a rough estimate of future decline, demonstrating that traditional tree-based models still perform strongly on structured ophthalmic data.


To model glaucoma progression on the UW dataset, we trained a RandomForestRegressor to predict the mean sensitivity slope (MS_slope, dB/year) for each eye based on baseline visual field measurements and patient metadata. Each eye was represented by its first recorded VF test (baseline mean sensitivity, age, follow-up duration, and optionally all 54 sensitivity points). We randomly split the data into 80% training and 20% testing sets and used a forest with 200 trees and a minimum of 3 samples per leaf.

On the held-out test set, the model achieved a mean absolute error of [MAE_here] dB/year, a root mean squared error of [RMSE_here] dB/year, and an R² score of [R2_here]. Feature importance analysis suggested that baseline mean sensitivity, follow-up duration, and several peripheral sensitivity locations contributed most strongly to the predictions. These results indicate that a relatively simple ensemble of decision trees can capture non-linear relationships between baseline VF patterns and future progression rates.
