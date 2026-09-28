# Credit Card Default Prediction & LASSO Regularization in R

## Overview
This repository contains an end-to-end predictive modeling project evaluating consumer credit default risk using high-dimensional financial data in RStudio.

![Executive Risk Dashboard](credit_default_executive_dashboard.png)

## Key Highlights
* **Data Preprocessing:** Cleaned 12,000+ billing records, converting categorical variables and splitting into 80/20 train/test sets.
* **Regularization & Feature Selection:** Applied K-fold cross-validated LASSO regression (`glmnet`) to manage collinearity among `PAY_*` and `BILL_AMT*` predictors.
* **Results:** Optimized penalty parameter $\lambda$ to reduce misclassification error below 19% while selecting the top 11 significant risk drivers.

## Repository Contents
* `credit_default_analysis.R` - Complete script containing data cleaning, modeling, and dashboard construction.
* `credit_default_executive_dashboard.png` - Rendered high-resolution summary dashboard.

## Packages Used
`tidyverse`, `dplyr`, `ggplot2`, `patchwork`, `glmnet`, `pROC`
