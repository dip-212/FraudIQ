# FraudIQ — Credit Card Fraud Detection

An end-to-end Machine Learning project that detects credit card fraud using a best-of-5 classifier pipeline, served via a Flask web application.

## Project Structure

```
deployment project/
├── app.py                  # Flask entry point
├── application.py          # AWS Elastic Beanstalk alias
├── setup.py                # Package installer
├── requirements.txt
│
├── src/
│   ├── exception.py        # Custom exception with traceback info
│   ├── logger.py           # Timestamped rotating logs
│   ├── utils.py            # save/load pickle, evaluate_models()
│   ├── components/
│   │   ├── data_ingestion.py       # CSV → stratified split → artifacts/
│   │   ├── data_transformation.py  # StandardScaler + SMOTE
│   │   └── model_trainer.py        # GridSearchCV → best model.pkl
│   └── pipeline/
│       ├── train_pipeline.py       # Full training orchestrator
│       └── predict_pipeline.py     # Inference: load pkl → predict
│
├── notebook/
│   └── eda_creditcard_fraud.ipynb  # EDA with visuals
│
├── templates/
│   ├── index.html          # Landing page
│   └── home.html           # Prediction form
│
├── artifacts/              # Auto-generated: data.csv, model.pkl, preprocessor.pkl
└── logs/                   # Auto-generated per run
```

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (takes ~10-15 min due to GridSearchCV)
python -m src.pipeline.train_pipeline

# 4. Run the Flask app
python app.py
# → Open http://localhost:5000
```

## Dataset

- **Source:** Kaggle Credit Card Fraud Detection dataset
- **Rows:** 284,807 transactions
- **Target:** `Class` (0=Legitimate, 1=Fraud)
- **Class balance:** 284,315 legit vs 492 fraud (0.17%)

## ML Pipeline

1. **Data Ingestion** — Stratified 80/20 split (preserves fraud ratio)
2. **Data Transformation** — StandardScaler on `Time` + `Amount`; V1–V28 pass through; SMOTE on train set
3. **Model Training** — GridSearchCV on 20% sample across 5 classifiers (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost); retrain winner on full data
4. **Selection Criterion** — ROC-AUC (required ≥ 0.85)

## Author

dip
