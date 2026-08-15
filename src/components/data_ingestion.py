import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    # Use pathlib for cross-platform path resolution
    source_data_path: str = str(
        Path(__file__).resolve().parent.parent.parent / "data" / "creditcard.csv"
    )


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            df = pd.read_csv(self.config.source_data_path)
            logging.info(f"Dataset loaded: shape={df.shape}")

            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved to {self.config.raw_data_path}")

            # Stratified split to preserve class balance in both sets
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42, stratify=df["Class"]
            )

            train_set.to_csv(self.config.train_data_path, index=False, header=True)
            test_set.to_csv(self.config.test_data_path, index=False, header=True)
            logging.info(
                f"Train set: {train_set.shape}, Test set: {test_set.shape}. "
                f"Fraud in train: {train_set['Class'].sum()}, "
                f"Fraud in test: {test_set['Class'].sum()}"
            )

            logging.info("Data ingestion completed")
            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)
