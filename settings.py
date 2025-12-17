
# app/settings.py
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_hgb.pkl")
BASELINE_STATS_PATH = os.path.join(ARTIFACTS_DIR, "baseline_stats.json")

# Monitoring thresholds (commonly used rules of thumb)
PSI_WARN = 0.1    # mild drift
PSI_ALERT = 0.2   # significant drift
KS_PVAL_WARN = 0.05  # KS p-value threshold (<= means distribution potentially changed)

# Rolling window for model drift evaluation (assumes you’ll receive labels later)
ROLLING_WINDOW = 500

FEATURE_ORDER = [
    "credit_score","previous_defaults","loan_term_months",
    "income_band_ord","loan_amount_band_ord","employment_length_band_ord","dti_band_ord",
    "gender_Female","gender_Male","age_band_18-30","age_band_31-59",
    "home_ownership_Family","home_ownership_Own","home_ownership_Rent"
]
