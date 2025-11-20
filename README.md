# Santander Customer Transaction Prediction  
### Unsupervised Learning Approach with EDA and Model Analysis

This project provides a complete unsupervised machine learning pipeline for the Santander Customer Transaction Prediction competition on Kaggle. The goal is to model customer behavior using only the feature distributions, without using labels during training, and then generate a submission file suitable for Kaggle.

The repository contains a notebook implementing exploratory data analysis, anomaly detection, validation analysis, and final submission generation.

---

## Overview

The competition challenges participants to predict whether a customer will make a transaction based on anonymized numeric features. This project uses an unsupervised anomaly detection strategy.  
The core hypothesis is that customers labeled as positive behave like rare events in the feature space, allowing anomaly detection techniques to capture unusual patterns.

The approach includes:

- Comprehensive exploratory data analysis  
- Unsupervised modeling using Isolation Forest  
- Score calibration and evaluation  
- ROC-AUC and precision-recall analysis  
- Final Kaggle submission file  

---

## Key Features

### Exploratory Data Analysis
- Distribution inspection of features  
- Target imbalance check  
- Correlation sampling  
- Comparison between train and test distributions  
- KDE plots contrasting feature behavior by class  

### Unsupervised Modeling
- Isolation Forest trained without labels  
- Scores inverted and min-max scaled to behave like probability estimates  
- Validation via ROC-AUC and precision-recall curves  
- Threshold search for model diagnostics

### Submission Pipeline
- Refit on the entire training set  
- Predict anomaly scores on the test set  
- Save predictions in the required `ID_code,target` format  

---

## Folder Structure

├── notebook.ipynb
├── submission_unsupervised_isoforest.csv
├── README.md
└── assets/ (optional: plots or images)

## Requirements

numpy
pandas
scikit-learn
matplotlib
seaborn

## Results

The notebook produces:
- Validation ROC-AUC
- Precision-recall curve analysis
- Distribution plots for anomaly scores
- A final submission file generated through an unsupervised anomaly detection process
- This approach is ideal for benchmarking, research, ensemble contributions, and experimentation with alternative modeling paradigms.
