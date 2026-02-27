# Credit Card Customer Churn Prediction

Predict which credit card customers are likely to close their accounts, then turn predictions into actionable retention strategy. Full pipeline from EDA through model export, with customer segmentation, threshold optimization, and cost-benefit analysis.

**[View the HTML Report](report.html)** - standalone report with all charts and tables, no notebook required.

## Dataset

[Credit Card Customers](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers) from Kaggle. 10,127 customers, 21 features.

Download `BankChurners.csv` and place it in the project root.

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/credit-card-churn-prediction.git
cd credit-card-churn-prediction

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook credit_card_churn_prediction.ipynb
```

## What's inside

**EDA** - Target distribution, churn rates by demographics / card type / income / age band, financial and behavioral feature distributions, inactivity and contact frequency risk views, correlation analysis.

**Customer Segmentation** - K-Means clustering with elbow + silhouette selection, churn rates per cluster, cluster profiling table with heatmap, feature distributions by segment.

**Dimensionality Reduction** - PCA and t-SNE 2D projections colored by churn status and cluster membership (discrete legends). Full explained variance spectrum.

**Preprocessing** - Ordinal encoding, standard scaling, SMOTE for class imbalance. Everything in sklearn/imblearn pipelines to prevent data leakage.

**Models** - Logistic Regression (baseline), Random Forest, Gradient Boosting, XGBoost, LightGBM. Stratified 5-fold CV on F1, ROC-AUC, precision, recall.

**Evaluation** - Test set metrics, confusion matrices for all five models, ROC curves, precision-recall curves, feature importance for RF / XGBoost / LightGBM.

**Threshold Tuning & Retention** - Precision/recall/F1 vs threshold plot, cost-benefit simulation with configurable assumptions, risk tier table for retention targeting.

**Export** - Best pipeline saved as `churn_model.pkl` via joblib.

## Results

| Metric | Score |
|--------|-------|
| Best model | LightGBM |
| F1 (test) | 0.91 |
| ROC-AUC (test) | 0.993 |
| Accuracy | 97% |
| Precision | 0.93 |
| Recall | 0.89 |

Top features: Total_Trans_Ct, Total_Trans_Amt, Total_Revolving_Bal, Total_Ct_Chng_Q4_Q1.

## Project structure

```
.
├── credit_card_churn_prediction.ipynb   # main notebook (run with outputs)
├── report.html                           # standalone HTML report
├── requirements.txt
├── README.md
├── .gitignore
└── BankChurners.csv                      # dataset (not tracked)
```

## Key findings

Transaction behaviour (count, amount, Q4/Q1 change) is the strongest churn signal by far. Clustering confirms this: the highest-churn segment has the lowest transaction activity and utilization. Engagement metrics (inactivity, bank contacts) are strong secondary signals. Demographics barely move the needle. LightGBM wins on F1 and ROC-AUC.

## License

MIT
