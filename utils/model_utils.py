import os
import pickle
from catboost import CatBoostClassifier

MODEL_PATH = "model/model.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    if not isinstance(model, CatBoostClassifier):
        raise TypeError("Loaded model is not a CatBoostClassifier")

    return model
