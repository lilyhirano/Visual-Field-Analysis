# Random Forest Regression – UW Visual Field Dataset

This folder contains a Random Forest model trained on the University of Washington Visual Field (UW VF) dataset to predict MS_Cluster3 from baseline visual field measurements.

# Overview

Each eye is represented by its first available visual field test, and we use baseline measurements such as global mean sensitivity (MS), cluster-level MS values, and pattern deviation (PD) features. The goal is to see how well these baseline signals can predict MS_Cluster3.

# Methods

We trained a RandomForestRegressor with:

200 trees

minimum leaf size of 3

80/20 train–test split

random seed = 42

Missing values in the feature matrix were filled using column means.

Results

<img width="2400" height="1800" alt="RFR" src="https://github.com/user-attachments/assets/2bb1f6d8-ed64-4085-9571-ed220546bb1c" />

The bar plot indicates that the Model's Mean Sensitivity (MS) is overwhelmingly important for the model's predictions, while nearly all of the Random Forest's importance weights are assigned to MS. All other features (MS clusters and individual PDs) provide relative little additional value to the model. This occurs because MS provides a global overview of the entire field of view. Since MS provides nearly all of the data needed to understand how damaged the eye is, the model was trained to predict MS. Therefore, the Forest simply learned that MS predicts MS, giving it an importance score of nearly 1.0.


<img width="4200" height="750" alt="image" src="https://github.com/user-attachments/assets/122c13bd-f885-4de2-b6f5-b675b7201fc8" />

Temperatures in the proximity temperature deviation heatmap are exclusively focused on the pattern deviation characteristics of the VFs and scale the influence of the characteristics against one another. While the amount of contribution made by each feature to the model is very small in absolute terms, the heatmap indicates which of those features have a higher relative impact than others. Examples of such features with greater relative importance include PD_36, PD_11, and PD_19. These points are often found in medically significant/good locations (nasal stepara, arecla zone), where, traditionally, glaucomatous changes are first found.

Hence, although the heatmap by itself is not necessarily predictive of future VF changes, it does identify which VFs are likely to be significant to a patient if their particular patient is not already affected by primary major conditions.

Overall, The global sensitivity score (MS) is the primary source of knowledge for the model, however, if only the PD features are assessed separately, we can see some geographic clustering of those VF locations that are contributing slightly more than others, which follows the traditional patterns of glandular damage in glaucoma.

# The model performed extremely well:

MAE: 0.0060

RMSE: 0.0375

R²: 1.0000

# Feature Importance

The feature importance plot showed that baseline `mean sensitivity` (MS) almost completely dominated the prediction. All other features—including PD points and MS cluster values—contributed very little.

This suggests that MS_Cluster3 is strongly determined by MS in this dataset, making the prediction task nearly redundant.

# Conclusion

The Random Forest achieved near-perfect accuracy, but mostly because MS_Cluster3 can be inferred almost directly from MS. Instead of revealing complex nonlinear patterns, the model exposed a strong internal relationship in the dataset. While the model technically performs well, the task may not represent true progression prediction.

A more meaningful next step would be to model future MS, change over time, or disease classification, where baseline features would play a more informative role.


# Choosing the Right Target: MS vs. MS_slope

How you use Mean Sensitivity (MS) depends on the goal of the analysis:

1. Using MS as the target (severity prediction)

This is the setting used in our current Random Forest model.
The goal is to predict the overall severity of the visual field from other VF features.
This could be useful if a clinic only had PD maps or cluster values and wanted to estimate MS.

However, because MS is also included in the input features, the model mostly learns to reproduce MS directly.
If MS is the prediction target, it should be removed from the feature list to avoid this identity effect.

2. Using MS_slope as the target (glaucoma progression)

A more clinically meaningful option is to predict MS_slope, which measures how quickly MS declines over time (dB/year).
This provides insight into which patients may progress faster and which VF regions are early indicators of future loss.

In this case, keeping baseline MS in the features is appropriate because a lower MS often correlates with faster decline, but MS itself is not identical to the slope.
Random Forest can then use MS along with the PD map and cluster features to learn non-linear patterns related to disease progression.
