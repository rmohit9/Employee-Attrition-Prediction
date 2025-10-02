# Employee Attrition Prediction

![Employee Attrition](https://img.shields.io/badge/Language-Python-blue) ![Status](https://img.shields.io/badge/Status-Completed-success)

## Overview

This project focuses on predicting employee attrition using machine learning techniques. Employee attrition refers to the voluntary or involuntary departure of employees from an organization, which can impact productivity and costs. By analyzing HR data, this model identifies key factors influencing attrition and builds predictive models to help organizations retain talent.

The analysis is performed in a Colab Notebook (`Employee_Attrition_Prediction.ipynb`), which includes data exploration, visualization, feature engineering, and model training/evaluation. The dataset used is the [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) (loaded as `Employee-Attrition.csv`).

### Key Objectives
- Explore relationships between employee attributes (e.g., age, job satisfaction, work-life balance).
- Visualize data distributions and correlations using KDE plots.
- Identify important features using Random Forest.
- Build and evaluate machine learning models for attrition prediction (e.g., Random Forest, Gradient Boosting, Logistic Regression, XGBoost).
- Handle class imbalance using SMOTE oversampling.

## Dataset

- **Source**: `Employee-Attrition.csv` (assumed to be the IBM HR Analytics dataset).
- **Features**: 35 attributes including:
  - Numerical: Age, DailyRate, DistanceFromHome, MonthlyIncome, TotalWorkingYears, YearsAtCompany, etc.
  - Categorical: BusinessTravel, Department, EducationField, Gender, JobRole, MaritalStatus, OverTime, etc.
- **Target Variable**: Attrition (Yes/No) – Binary classification problem.
- **Size**: Approximately 1470 records.
- **Class Imbalance**: Attrition cases are typically underrepresented (~16% Yes).

## Requirements

To run the notebook, install the following Python libraries:

```bash
pip install numpy pandas seaborn matplotlib plotly scikit-learn imbalanced-learn xgboost