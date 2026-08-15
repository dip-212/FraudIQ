import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray) -> float:
        try:
            logging.info("Splitting train/test arrays for model training")
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # Sample 10% of SMOTE-augmented train data for fast, accurate GridSearchCV,
            # then retrain best model on the training set
            from sklearn.model_selection import train_test_split
            X_sample, _, y_sample, _ = train_test_split(
                X_train, y_train, train_size=0.10, random_state=42, stratify=y_train
            )
            logging.info(
                f"GridSearchCV sample size: {X_sample.shape[0]} rows "
                f"(from {X_train.shape[0]} SMOTE-augmented rows)"
            )
            print(f"[*] Comparing 5 models on {X_sample.shape[0]:,} SMOTE samples with 3-fold CV ...")

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "XGBoost": XGBClassifier(
                    eval_metric="logloss",
                    random_state=42,
                    verbosity=0,
                    n_jobs=-1,
                ),
            }

            params = {
                "Logistic Regression": {
                    "C": [0.1, 1.0, 10.0],
                    "solver": ["lbfgs"],
                },
                "Decision Tree": {
                    "criterion": ["gini"],
                    "max_depth": [5, 10, None],
                },
                "Random Forest": {
                    "n_estimators": [50, 100],
                    "max_depth": [10, 20],
                },
                "Gradient Boosting": {
                    "n_estimators": [50],
                    "learning_rate": [0.1],
                    "max_depth": [3, 5],
                },
                "XGBoost": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.1],
                    "max_depth": [3, 6],
                },
            }

            logging.info("Running GridSearchCV on sample for each model …")
            model_report, fitted_models = evaluate_models(
                X_train=X_sample,
                X_test=X_test,
                y_train=y_sample,
                y_test=y_test,
                models=models,
                params=params,
            )
            logging.info(f"Model ROC-AUC report: {model_report}")
            print(f"\n[*] Model evaluation report:\n{model_report}")

            # Select best model by ROC-AUC
            best_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)

            if best_score < 0.85:
                raise CustomException(
                    f"No model achieved ROC-AUC >= 0.85. Best was {best_model_name}: {best_score:.4f}",
                    sys,
                )

            logging.info(
                f"Best model selected: {best_model_name} (ROC-AUC: {best_score:.4f}). "
                "Retraining final estimator on full training set …"
            )
            print(f"\n[+] Winner: {best_model_name} (Sample ROC-AUC: {best_score:.4f})")
            print(f"[+] Retraining {best_model_name} on full dataset ...")

            # Retrain best model on FULL SMOTE-augmented training data
            best_model_class = type(fitted_models[best_model_name])
            best_params = fitted_models[best_model_name].get_params()
            final_model = best_model_class(**best_params)
            final_model.fit(X_train, y_train)

            # Evaluate final model on unseen test set
            y_prob = final_model.predict_proba(X_test)[:, 1]
            final_roc_auc = roc_auc_score(y_test, y_prob)

            y_pred = final_model.predict(X_test)
            final_f1 = f1_score(y_test, y_pred)

            logging.info(f"Final model test ROC-AUC: {final_roc_auc:.4f}")
            logging.info(f"Final model test F1-score: {final_f1:.4f}")
            logging.info(f"Classification report:\n{classification_report(y_test, y_pred)}")

            save_object(file_path=self.config.model_path, obj=final_model)
            logging.info(f"Best model saved to {self.config.model_path}")

            return final_roc_auc

        except Exception as e:
            raise CustomException(e, sys)
