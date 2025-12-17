
# app/schemas.py
from typing import List, Optional
from pydantic import BaseModel, Field

class BorrowerFeatures(BaseModel):
    credit_score: float
    previous_defaults: float
    loan_term_months: float
    income_band_ord: float
    loan_amount_band_ord: float
    employment_length_band_ord: float
    dti_band_ord: float
    gender_Female: float
    gender_Male: float
    age_band_18-30: float = Field(alias="age_band_18_30")
    age_band_31-59: float = Field(alias="age_band_31_59")
    home_ownership_Family: float
    home_ownership_Own: float
    home_ownership_Rent: float

    class Config:
        populate_by_name = True  # allows using age_band_18_30 in JSON

class PredictRequest(BaseModel):
    records: List[BorrowerFeatures]

class PredictResponse(BaseModel):
    probabilities: List[float]            # P(Default)
    predictions: List[int]                 # 0=Paid/Current, 1=Default
    threshold: float = 0.5
