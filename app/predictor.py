from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR /
    "models/agrisense_yield_model_v1.pkl"
)


class YieldPredictor:

    def __init__(self):

        if not MODEL_PATH.exists():

            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        self.model = joblib.load(
            MODEL_PATH
        )

    def predict(self, features):

        prediction = self.model.predict(
            features
        )

        return float(prediction[0])


predictor = YieldPredictor()
