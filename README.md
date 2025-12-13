# Chem 277B — Machine Learning Algorithms
## Convolutional Neural Network Analysis of Visual Field Maps for Early Diagnosis and Prediction of Glaucoma Progression
Team 1 — UC Berkeley, College of Chemistry

Contributors: David Houshangi, Lily Hirano, Kirk Ehmsen, Christian Fernandez, Yash Maheshwaran

# 1. Project Overview

Glaucoma is a chronic, progressive disease that damages the optic nerve and can lead to irreversible blindness if not detected early. Because early stages often have no symptoms, nearly half of affected individuals do not know they have the disease. Visual field (VF) testing is one of the primary tools for diagnosing glaucoma and tracking its progression, as it measures the patient’s functional vision over time.

This project explores how modern machine-learning methods, including unsupervised learning, Random Forests, CNNs, and LSTMs can be used to analyze visual field maps, detect patterns of glaucomatous loss, and predict future progression.

We used two major datasets:

1. GRAPE Dataset: 263 eyes, 1,115 longitudinal VF tests

2. UW Biomedical AI Dataset: 3,871 patients, 28,943 VF tests

Together, these datasets allow us to study glaucoma progression using real functional measurements rather than structural imaging alone.

# 2. Project Goals

**2.1 Unsupervised Learning**

Cluster visual fields into meaningful glaucoma subtypes based on the spatial pattern and rate of deterioration.

**2.2 Gradient Boost Regression**

Predict long-term progression (MS slope) from baseline, early-window, and MS acceleration features. Identify the strongest feature predictors of decline.

**2.3 Random Forest Regression**

Predict long-term progression (MS slope) from baseline VF features and identify the strongest physiological predictors of decline.

**2.4 CNN-Based VF Classification**

Train convolutional neural networks to classify VF maps into severity categories by learning spatial patterns of damage.

**2.5 LSTM-Based Progression Modeling**

Use longitudinal VF sequences to model temporal dynamics and predict how damage evolves over time.

**2.6 Dataset Alignment**

Standardize map formats so models trained on UW data can generalize to GRAPE, enabling cross-dataset comparisons.

# 3. Repository Structure






# 4. Methods 

- Unsupervised Learning: PCA + KMeans + UMAP for clustering VF progression patterns

- Gradient Boosting: Predict slope of MS loss; extract feature importances

- Random Forest Regression: Predict slope of MS loss; extract feature importances

- CNN Models: Classify VF severity from interpolated VF maps

- LSTM Models: Sequence-based prediction of future sensitivity loss


# 5. References

1. GRAPE Dataset
Huang et al. GRAPE: A multi-modal dataset of longitudinal follow-up visual fields and fundus images for glaucoma management.

2. UW Biomedical AI Dataset
Wang et al. A large-scale clinical visual field database for glaucoma analysis.
