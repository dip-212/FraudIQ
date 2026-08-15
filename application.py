import os
import sys

from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import PredictPipeline, TransactionData
from src.exception import CustomException
from src.logger import logging

application = Flask(__name__)
app = application

# ── Load model & preprocessor ONCE at startup (not per request) ──────────────
MODEL_PATH = os.path.join("artifacts", "model.pkl")
PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.pkl")

_pipeline = None

def get_pipeline() -> PredictPipeline:
    """Lazy-load and cache the pipeline on first request."""
    global _pipeline
    if _pipeline is None:
        logging.info("Loading model and preprocessor into memory …")
        _pipeline = PredictPipeline(
            model_path=MODEL_PATH,
            preprocessor_path=PREPROCESSOR_PATH,
        )
        logging.info("Pipeline ready.")
    return _pipeline


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")

    try:
        form = request.form

        # Input validation
        def get_float(key: str) -> float:
            val = form.get(key, "").strip()
            if val == "":
                raise ValueError(f"Field '{key}' is required.")
            return float(val)

        data = TransactionData(
            time=get_float("time"),
            amount=get_float("amount"),
            v1=get_float("v1"), v2=get_float("v2"), v3=get_float("v3"), v4=get_float("v4"),
            v5=get_float("v5"), v6=get_float("v6"), v7=get_float("v7"), v8=get_float("v8"),
            v9=get_float("v9"), v10=get_float("v10"), v11=get_float("v11"), v12=get_float("v12"),
            v13=get_float("v13"), v14=get_float("v14"),
        )

        df = data.get_data_as_dataframe()
        pipeline = get_pipeline()
        label, probability = pipeline.predict(df)

        result = {
            "label": label,              # 0 or 1
            "probability": round(probability * 100, 2),  # fraud %
            "verdict": "FRAUD" if label == 1 else "LEGITIMATE",
        }

        logging.info(f"Prediction: {result}")
        return render_template("home.html", result=result)

    except ValueError as ve:
        return render_template("home.html", error=str(ve))
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False)
