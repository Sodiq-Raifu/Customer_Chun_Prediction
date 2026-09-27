# Telco Customer Churn Analysis

Exploratory data analysis, survival analysis, and predictive modeling on the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset, aimed at understanding *why* and *when* customers churn, and predicting *who* is likely to churn next.

## Contents

- **Data Cleaning** – handling `TotalCharges` type conversion, missing values, and duplicates.
- **Exploratory Data Analysis (EDA)** – payment method distribution, churn by gender, correlation heatmap, and relationships between tenure, monthly charges, total charges, and churn.
- **Survival Analysis** – Kaplan-Meier curves (overall, by contract type, and by internet service) and a Cox Proportional Hazards model to estimate customer "time-to-churn" and the effect of different features on churn risk.
- **Predictive Modeling** – one-hot encoding of categorical features, train/test split, and comparison of three classifiers:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - XGBoost

## Key Findings

- Electronic check is the most common payment method (~2,365 users).
- Customers paying premium monthly rates ($60–$100) churn at a noticeably higher rate.
- Long-tenured customers are far less likely to churn than newer customers.
- Model performance on the test set:

| Model | Accuracy | Precision (Churn) | Recall (Churn) |
|---|---|---|---|
| Logistic Regression | 0.75 | 0.53 | 0.83 |
| KNN | 0.78 | 0.61 | 0.55 |
| XGBoost | 0.79 | 0.64 | 0.52 |

XGBoost gives the best overall accuracy, while Logistic Regression catches the most churners (highest recall) at the cost of more false positives — worth considering depending on whether the business prioritizes catching churners or minimizing false alarms.

## Dataset

This notebook expects the `WA_Fn-UseC_-Telco-Customer-Churn.csv` file (IBM's public Telco Customer Churn dataset). You can download it from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in the project directory.

> **Note:** The notebook currently reads the file from a local path (`C:\Users\...`). Update the `pd.read_csv(...)` path in the second cell to point to wherever you've saved the CSV, e.g.:
> ```python
> df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
> ```

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lifelines
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lifelines
```

## Usage

1. Clone this repo and install the requirements above.
2. Download the dataset and update the file path in the notebook.
3. Run `Churn_Analysis_Notebook.ipynb` cell by cell in Jupyter.

## License

Feel free to use or adapt this analysis for learning purposes.
