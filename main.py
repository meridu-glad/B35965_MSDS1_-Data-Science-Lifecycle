# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import pandas as pd
# No other imports needed based on this snippet

# Define the FastAPI app instance here
app = FastAPI(title="Loan Default Prediction API")

# Path setup - Corrected
model_dir = "models" 
# Ensure the 'models' folder exists and 'loan_model.pkl' is inside it
model_path = os.path.join(model_dir, "loan_model.pkl")
# Assuming the model file path is correct in your environment:
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    # You might want to exit the application or handle this gracefully

class LoanFeatures(BaseModel):
    # Update these names to match EXACTLY what your model expects
    age_band_18_30: float  # Changed '-' to '_'
    age_band_31_59: float
    credit_score: float
    credit_score_scaled: float
    dti_band_ord: float
    # ... ensure all other missing features from the error log are listed here ...

@app.post("/predict")
def predict_loan_status(data: LoanFeatures):
    # Use .model_dump() for Pydantic v2 (standard in 2025)
    input_df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(input_df)

    # Result mapping
    status = "Default" if prediction[0] == 1 else "Paid/Current"
    return {"prediction": status}

