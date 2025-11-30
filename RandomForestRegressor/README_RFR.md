Random Forest Regression – UW Visual Field Dataset

This folder contains a Random Forest model trained on the University of Washington Visual Field (UW VF) dataset to predict MS_Cluster3 from baseline visual field measurements.

Overview

Each eye is represented by its first available visual field test, and we use baseline measurements such as global mean sensitivity (MS), cluster-level MS values, and pattern deviation (PD) features. The goal is to see how well these baseline signals can predict MS_Cluster3.

Methods

We trained a RandomForestRegressor with:

200 trees

minimum leaf size of 3

80/20 train–test split

random seed = 42

Missing values in the feature matrix were filled using column means.

Results

The model performed extremely well:

MAE: 0.0060

RMSE: 0.0375

R²: 1.0000

Feature Importance

The feature importance plot showed that baseline mean sensitivity (MS) almost completely dominated the prediction. All other features—including PD points and MS cluster values—contributed very little.

This suggests that MS_Cluster3 is strongly determined by MS in this dataset, making the prediction task nearly redundant.

Conclusion

The Random Forest achieved near-perfect accuracy, but mostly because MS_Cluster3 can be inferred almost directly from MS. Instead of revealing complex nonlinear patterns, the model exposed a strong internal relationship in the dataset. While the model technically performs well, the task may not represent true progression prediction.

A more meaningful next step would be to model future MS, change over time, or disease classification, where baseline features would play a more informative role.
