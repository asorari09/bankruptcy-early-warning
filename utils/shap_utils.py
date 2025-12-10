import shap
import numpy as np

def load_shap_explainer(model):
    """SHAP TreeExplainer for CatBoost."""
    try:
        explainer = shap.TreeExplainer(model)
    except Exception as e:
        raise RuntimeError(f"Failed to create SHAP explainer: {e}")
    return explainer

def compute_local_shap(explainer, features_df):
    """
    Compute SHAP values with compatibility across SHAP versions:
    - shap_values may be a list (multiclass) or ndarray.
    - expected_value may be scalar or list-like.
    """
    raw_shap = explainer.shap_values(features_df)
    if isinstance(raw_shap, list):
        shap_values = raw_shap[0]
    else:
        shap_values = raw_shap

    # Take first row for single-instance input
    shap_row = shap_values[0] if getattr(shap_values, "ndim", 1) > 1 else shap_values

    base_raw = explainer.expected_value
    if isinstance(base_raw, (list, tuple, np.ndarray)):
        base_value = float(base_raw[0])
    else:
        base_value = float(base_raw)

    return shap_row, base_value

def get_top_shap_features(shap_values, feature_names, top_n=8):
    arr = np.array(shap_values)
    idx = np.argsort(np.abs(arr))[-top_n:][::-1]
    return [
        {"feature": feature_names[i], "shap_value": float(arr[i])}
        for i in idx
    ]
