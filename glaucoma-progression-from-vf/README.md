# Predicting Glaucoma Severity and Progression from Humphrey Visual Field (VF) Data

This project explores whether deep learning and classical machine-learning models can detect visual field damage and predict future progression of glaucoma using the UW Humphrey Visual Field (UWHVF) dataset. The dataset contains thousands of real clinical 24-2 visual field tests collected longitudinally across many patients. Each VF test includes numerical sensitivity values at 54 test points, global indices (MTD, PSD, MS), patient attributes, and follow-up time.

The goal of this project is to transform these VF measurements into 2D images, train models to recognize glaucomatous patterns, and forecast the likely rate of visual field loss over time. In the long term, this pipeline will support a simple GUI where users can upload a VF image and receive both a severity estimate and a predicted progression rate.
