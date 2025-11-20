# Chem 277B - Machine Learning Algorithms  
## Convolutional Neural Network Analysis of Visual Field Maps for Early Diagnosis and Prediction of Glaucoma Progression
Team 1 — UC Berkeley, College of Chemistry

Contributors: David Houshangi, Lily Hirano, Kirk Ehmsen, Yash Maheshwaran, Christian Fernandez

> Subject: Visual-Field-Analysis

# 1. Overview

Glaucoma is a progressive disease that gradually damages the optic nerve and can lead to permanent blindness if not detected early. Nearly half of affected individuals are unaware they have it because early-stage glaucoma often presents no symptoms. Visual field (VF) testing, which measures functional vision loss, is one of the most important tools for glaucoma diagnosis and monitoring.

This project explores how Convolutional Neural Networks (CNNs) and supervised learning can be applied to:

- Analyze visual field maps

- Identify patterns of glaucomatous vision loss

- Cluster progression subtypes

- Predict future deterioration

- Compare disease severity across datasets

We use two major datasets:

- GRAPE Dataset – Longitudinal VF tests (263 eyes, 1,115 follow-ups)

- UW Biomedical AI Dataset – Large-scale VF dataset (3,871 patients, 28,943 tests)

These datasets provide real-world measurements of visual sensitivity, allowing us to examine glaucoma progression through functional vision rather than structural imaging alone.

# 2. Project Goals

Our project focuses on the following objectives:

## 2.1 Glaucoma Subtype Discovery

Use unsupervised learning to cluster glaucoma progression patterns based on spatial deterioration of visual fields.

## 2.2 CNN-Based Vision Loss Classification

Train CNN models to classify VF maps into disease severity categories using spatial patterns, color maps, and sensitivity distributions.

## 2.3 Progression Prediction

Use longitudinal fields from GRAPE to estimate:

- Rate of vision decline

- Risk factors (age, sex, subtype)

- Likelihood of rapid or slow progression

## 2.4 Dataset Alignment

Standardize VF map formats across the two datasets so that a model trained on one dataset can generalize to the other.

# Repository Structure

Visual-Field-Analysis/
│
├── data/                       
│   ├── grape/                 
│   ├── uw/                     
│   ├── Coord_242.csv           
│   └── Add here                   
│
├── images/                     
│   ├── grape_vf_example.png
│   ├── uw_vf_example.png
│   └── interpolated_vf_maps/
│
├── references/                 
│   ├── GRAPE_paper.pdf
│   ├── UWHVF_paper.pdf
│   └── bibliography.txt
│
├── eda_glaucoma_shared.ipynb   
├── grape_images.ipynb          
├── uw_images.ipynb             
│
├── Makefile                    
├── requirements.txt            
├── README.md                  
│
└── .gitignore                  


## References
- GRAPE Dataset
    - [GRAPE: A multi-modal dataset of longitudinal follow-up visual field and fundus images for glaucoma management](https://www.nature.com/articles/s41597-023-02424-4)
    - [GRAPE Dataset](https://springernature.figshare.com/collections/GRAPE_A_multi-modal_glaucoma_dataset_of_follow-up_visual_field_and_fundus_images_for_glaucoma_management/6406319/1)

- UWHVF
    - [UW Biomedical VF GitHub](https://github.com/uw-biomedical-ml/uwhvf)
