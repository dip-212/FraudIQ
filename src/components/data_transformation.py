import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    """
    Scales Time and Amount (V1-V28 are already PCA-scaled).
    Applies SMOTE on training data only to handle class imbalance.
    """

    # Features that need scaling - V1-V28 are already PCA-transformed
    SCALE_COLS = ["Time", "Amount"]

    # Top PCA features exposed in the UI (V1-V14)
    V_FEATURES = [f"V{i}" for i in range(1, 15)]

    # All features used for training (full set)
    ALL_FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

    def __init__(self):
        self.config = DataTransformationConfig()

    def get_preprocessor(self):
        """
        Build a ColumnTransformer that scales Time and Amount;
        V1-V28 pass through unchanged.
        """
        try:
            passthrough_cols = [f"V{i}" for i in range(1, 29)]

            scale_pipeline = Pipeline(steps=[("scaler", StandardScaler())])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("scale", scale_pipeline, self.SCALE_COLS),
                    ("passthrough", "passthrough", passthrough_cols),
                ],
                remainder="drop",
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Read train/test CSVs → fit preprocessor on train →
        apply SMOTE on train array → return arrays + preprocessor path.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

            target_col = "Class"

            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]
            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]

            preprocessor = self.get_preprocessor()
            logging.info("Fitting preprocessor on training data")

            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)

            logging.info(
                f"Before SMOTE — class distribution: "
                f"{dict(zip(*np.unique(y_train, return_counts=True)))}"
            )

            # Apply SMOTE only on training set (prevents data leakage)
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(
                X_train_scaled, y_train
            )
            logging.info(
                f"After SMOTE - class distribution: "
                f"{dict(zip(*np.unique(y_train_resampled, return_counts=True)))}"
            )

            train_arr = np.c_[X_train_resampled, np.array(y_train_resampled)]
            test_arr = np.c_[X_test_scaled, np.array(y_test)]

            save_object(
                file_path=self.config.preprocessor_file_path,
                obj=preprocessor,
            )
            logging.info(f"Preprocessor saved to {self.config.preprocessor_file_path}")

            return train_arr, test_arr, self.config.preprocessor_file_path

        except Exception as e:
            raise CustomException(e, sys)
