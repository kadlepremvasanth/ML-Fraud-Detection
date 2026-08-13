# Fraud Detection using Machine Learning

A machine learning project that detects fraudulent financial transactions using the PaySim synthetic mobile-money dataset. The pipeline covers data exploration, feature engineering, and a Logistic Regression classifier wrapped in a scikit-learn `Pipeline`.

##  Overview

Financial fraud is rare compared to legitimate transactions, which makes it a highly imbalanced classification problem. This project builds an end-to-end pipeline to flag fraudulent transactions (`isFraud`) from transaction-level data such as amount, transaction type, and account balances.

## Dataset

- **File:** `AIML Dataset.csv` (PaySim-style synthetic transaction data)
- **Rows:** ~6.36 million transactions
- **Target column:** `isFraud` (0 = legitimate, 1 = fraudulent)
- **Class imbalance:** only ~0.13% of transactions are fraudulent

**Key columns:**

| Column | Description |
|---|---|
| `step` | Time step (hour) of the transaction |
| `type` | Transaction type (PAYMENT, TRANSFER, CASH_OUT, etc.) |
| `amount` | Transaction amount |
| `nameOrig` / `nameDest` | Origin / destination account IDs |
| `oldbalanceOrg` / `newbalanceOrig` | Sender's balance before / after |
| `oldbalanceDest` / `newbalanceDest` | Receiver's balance before / after |
| `isFraud` | Target label |
| `isFlaggedFraud` | Flag from the original rule-based system |

##  Workflow

1. **Data loading & inspection** — checked shape, dtypes, missing values, and class balance.
2. **Feature engineering** — derived two new features to capture inconsistencies in balances:
   - `balanceDiffOrig = oldbalanceOrg - newbalanceOrig`
   - `balanceDiffDest = newbalanceDest - oldbalanceDest`
3. **Preprocessing** — dropped identifier/leak-prone columns (`step`, `nameOrig`, `nameDest`, `isFlaggedFraud`); split numeric vs categorical features.
4. **Pipeline** — built with `ColumnTransformer`:
   - `StandardScaler` on numeric features (`amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`)
   - `OneHotEncoder` on the categorical `type` feature
5. **Model** — `LogisticRegression(class_weight="balanced", max_iter=1000)` to account for class imbalance.
6. **Train/test split** — 70/30 stratified split (`random_state=42`).
7. **Evaluation** — classification report, confusion matrix, and accuracy score.
8. **Model export** — trained pipeline saved with `joblib` as `fraud_detection_pipeline.pkl`.

##  Results

| Metric | Class 0 (Legit) | Class 1 (Fraud) |
|---|---|---|
| Precision | 1.00 | 0.02 |
| Recall | 0.95 | 0.94 |
| F1-score | 0.97 | 0.04 |

- **Overall accuracy:** ~94.67%
- **Recall on fraud class:** ~94% — the model catches most fraudulent transactions, at the cost of many false positives (expected trade-off with `class_weight="balanced"` on a highly imbalanced dataset).

##  Tech Stack

- Python
- pandas, numpy
- matplotlib, seaborn (EDA/visualization)
- scikit-learn (`Pipeline`, `ColumnTransformer`, `LogisticRegression`, `StandardScaler`, `OneHotEncoder`)
- joblib (model persistence)

##  Getting Started

```bash
git clone <your-repo-url>
cd fraud-detection-ml
pip install pandas numpy matplotlib seaborn scikit-learn joblib
jupyter notebook Fraud_Detection_ML.ipynb
```

Place `AIML Dataset.csv` in the project directory before running the notebook.

##  Project Structure

```
├── Fraud_Detection_ML.ipynb       # Main notebook: EDA, feature engineering, modeling
├── AIML Dataset.csv               # Dataset (not included — add your own)
├── fraud_detection_pipeline.pkl   # Saved trained pipeline
└── README.md
```
