# 💳 Credit Card Fraud Detection with Imbalanced Data Handling

A machine learning project focused on detecting fraudulent credit card transactions using statistical preprocessing, advanced resampling techniques, and classification algorithms. This project emphasizes real-world challenges such as imbalanced data and evaluation beyond accuracy.

## 📌 Project Overview

Dataset: Public dataset from Kaggle containing credit card transactions from European cardholders in September 2013.

Goal: Automatically identify fraudulent transactions using classification models, while handling severe data imbalance (fraud rate ≈ 0.17%).

## 📂 Features

Total: 30 numerical features (28 anonymized PCA components + Time + Amount)

Target variable: Class (1 = Fraud, 0 = Normal)

## ⚙️ Preprocessing Pipeline

Feature Selection: Dropped Time and standardized Amount due to scale disparity.

Standardization: Applied StandardScaler to all features for normalized input to ML models.

## ⚖️ Imbalanced Data Techniques

Implemented multiple sampling methods to address class imbalance:

🧹 Random Under-Sampling

🎯 NearMiss (v1, v2, v3)

🔁 SMOTE (Synthetic Minority Over-sampling Technique)

🧪 SMOTE + Tomek Links (Hybrid sampling)

## 🤖 Machine Learning Models

Two models were applied and optimized using Grid Search + Cross-Validation:

Logistic Regression (with L1 and L2 regularization)

Random Forest Classifier


## 🧪 Model Evaluation Metrics

Standard accuracy is misleading on imbalanced data, so the following were used:

Recall (Fraud detection rate)

Precision

F1-Score

ROC-AUC

Confusion Matrix

## 📎 Tools Used

Python

scikit-learn

imbalanced-learn

matplotlib

seaborn


## 🧠 Lessons Learned

Class imbalance severely distorts common metrics like accuracy.

SMOTE and hybrid resampling significantly improve recall.

Ensemble models like Random Forest offer better generalization on such noisy and skewed data.

## 📜 Reference

Kaggle Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

SMOTE & Imbalanced-Learn Library

Scikit-learn documentation
