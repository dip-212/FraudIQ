import os
import sys
import pickle

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging


def save_object(file_path: str, obj) -> None:
    """Serialize an object to disk using pickle."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    """Deserialize an object from disk using pickle."""
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, X_test, y_train, y_test, models: dict, params: dict) -> tuple:
    """
    Run GridSearchCV (cv=3) for each model and evaluate on test set using ROC-AUC.
    
    Returns:
        report (dict): model_name → test ROC-AUC score
        fitted_models (dict): model_name → best fitted estimator
    """
    try:
        report = {}
        fitted_models = {}

        for name, model in models.items():
            param_grid = params.get(name, {})
            logging.info(f"--> Training & tuning {name} ...")
            print(f"--> Training & tuning {name} ...")
            gs = GridSearchCV(
                model,
                param_grid,
                cv=3,
                scoring="roc_auc",
                n_jobs=-1,
                verbose=0,
            )
            gs.fit(X_train, y_train)

            best_estimator = gs.best_estimator_
            fitted_models[name] = best_estimator

            y_prob = best_estimator.predict_proba(X_test)[:, 1]
            test_score = roc_auc_score(y_test, y_prob)
            report[name] = test_score
            logging.info(f"    [OK] {name} ROC-AUC: {test_score:.4f} (best params: {gs.best_params_})")
            print(f"    [OK] {name} ROC-AUC: {test_score:.4f}")

        return report, fitted_models

    except Exception as e:
        raise CustomException(e, sys)
