"""
train_pipeline.py — Orchestrates the full training workflow.

Run with:
    python -m src.pipeline.train_pipeline
"""
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def run_training_pipeline():
    try:
        logging.info("=" * 60)
        logging.info("TRAINING PIPELINE STARTED")
        logging.info("=" * 60)

        # Step 1 — Data Ingestion
        logging.info("Step 1: Data Ingestion")
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

        # Step 2 — Data Transformation
        logging.info("Step 2: Data Transformation + SMOTE")
        transformation = DataTransformation()
        train_arr, test_arr, _ = transformation.initiate_data_transformation(
            train_path, test_path
        )

        # Step 3 — Model Training
        logging.info("Step 3: Model Training + GridSearchCV")
        trainer = ModelTrainer()
        roc_auc = trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info("=" * 60)
        logging.info(f"TRAINING PIPELINE COMPLETE — Final ROC-AUC: {roc_auc:.4f}")
        logging.info("=" * 60)
        return roc_auc

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    score = run_training_pipeline()
    print(f"\n[SUCCESS] Training complete. Final ROC-AUC on test set: {score:.4f}")
