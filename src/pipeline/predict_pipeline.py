import sys
import os
from dataclasses import dataclass

import pandas as pd
import numpy as np

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    """
    Loads the saved preprocessor and model once (at import time)
    and reuses them for every prediction — no per-request disk I/O.
    """

    def __init__(self, model_path: str, preprocessor_path: str):
        try:
            self.model = load_object(model_path)
            self.preprocessor = load_object(preprocessor_path)
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features: pd.DataFrame):
        """
        Transform input features and return (label, probability_of_fraud).
        """
        try:
            scaled = self.preprocessor.transform(features)
            prediction = self.model.predict(scaled)
            probability = self.model.predict_proba(scaled)[:, 1]
            return int(prediction[0]), float(probability[0])
        except Exception as e:
            raise CustomException(e, sys)


@dataclass
class TransactionData:
    """
    Data-transfer object that maps form fields to a DataFrame row
    compatible with the trained preprocessor.
    """
    time: float
    amount: float
    v1: float; v2: float; v3: float; v4: float
    v5: float; v6: float; v7: float; v8: float
    v9: float; v10: float; v11: float; v12: float
    v13: float; v14: float

    # V15–V28 default to 0.0 (not exposed in the simplified UI form)
    v15: float = 0.0; v16: float = 0.0; v17: float = 0.0; v18: float = 0.0
    v19: float = 0.0; v20: float = 0.0; v21: float = 0.0; v22: float = 0.0
    v23: float = 0.0; v24: float = 0.0; v25: float = 0.0; v26: float = 0.0
    v27: float = 0.0; v28: float = 0.0

    def get_data_as_dataframe(self) -> pd.DataFrame:
        """Return a single-row DataFrame matching the model's expected feature order."""
        try:
            row = {
                "Time": [self.time],
                "V1": [self.v1], "V2": [self.v2], "V3": [self.v3], "V4": [self.v4],
                "V5": [self.v5], "V6": [self.v6], "V7": [self.v7], "V8": [self.v8],
                "V9": [self.v9], "V10": [self.v10], "V11": [self.v11], "V12": [self.v12],
                "V13": [self.v13], "V14": [self.v14],
                "V15": [self.v15], "V16": [self.v16], "V17": [self.v17], "V18": [self.v18],
                "V19": [self.v19], "V20": [self.v20], "V21": [self.v21], "V22": [self.v22],
                "V23": [self.v23], "V24": [self.v24], "V25": [self.v25], "V26": [self.v26],
                "V27": [self.v27], "V28": [self.v28],
                "Amount": [self.amount],
            }
            return pd.DataFrame(row)
        except Exception as e:
            raise CustomException(e, sys)
