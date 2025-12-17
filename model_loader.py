
# app/model_loader.py
import joblib
import numpy as np
import pandas as pd
from .settings import MODEL_PATH, FEATURE_ORDER

_model = None

def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

def to_dataframe(records):
    """records: list[dict] with keys matching FEATURE_ORDER"""
    df = pd.DataFrame(records)
    # Ensure column order, set dtype to float where appropriate
    df = df.reindex(columns=FEATURE_ORDER)
    # Convert numeric-like columns to float for robust inference and PDP compatibility
    float_cols = [
        "credit_score","previous_defaults","loan_term_months",
        "income_band_ord","loan_amount_band_ord","employment_length_band_ord","dti_band_ord",
        "gender_Female","gender_Male","age_band_18-30","age_band_31-59",
        "home_ownership_Family","home_ownership_Own","home_ownership_Rent"
    ]
    df[float_cols] = df[float_cols].astype("float32")
    return df
