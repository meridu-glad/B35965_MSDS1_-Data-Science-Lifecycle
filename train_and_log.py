
# train_and_log.py
import os
import pandas as pd
import numpy as np
import joblib
import mlflow

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

# --- Resolve directories based on *this file's* location
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# --- Data path: moses.csv should be in the same folder as this script
DATA_PATH = os.path.join(PROJECT_ROOT, "moses.csv")   # adjust if your CSV is elsewhere
CSV_PATH   = os.path.join(ARTIFACTS_DIR, "experiments_log.csv")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_hgb.pkl")

# --- Load data
df = pd.read_csv(DATA_PATH)
y = (df["loan_status"] == "Default").astype(int)
features = [
    'credit_score','previous_defaults','loan_term_months',
    'income_band_ord','loan_amount_band_ord','employment_length_band_ord','dti_band_ord',
    'gender_Female','gender_Male','age_band_18-30','age_band_31-59',
    'home_ownership_Family','home_ownership_Own','home_ownership_Rent'
]
X = df[features].copy()

# --- Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# --- Model & random search
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
hgb = HistGradientBoostingClassifier(random_state=42)
hgb_dist = {
    'learning_rate': [0.03, 0.05, 0.1],
    'max_depth': [None, 6, 8, 12],
    'max_leaf_nodes': [15, 31, 63],
    'min_samples_leaf': [10, 20, 30]
}
hgb_rs = RandomizedSearchCV(
    hgb, hgb_dist, cv=cv, scoring='roc_auc', n_iter=20, random_state=42, n_jobs=-1
)
hgb_rs.fit(X_train, y_train)
best_hgb = hgb_rs.best_estimator_

# --- Metrics
prob_test = best_hgb.predict_proba(X_test)[:, 1]
roc_test  = roc_auc_score(y_test, prob_test)
pr_test   = average_precision_score(y_test, prob_test)

# --- Save artifacts locally first
joblib.dump(best_hgb, MODEL_PATH)
pd.DataFrame([{
    "model":"HistGradientBoosting",
    "roc_auc_val": hgb_rs.best_score_,
    "pr_auc_val": np.nan,      # (optional) fill if you compute CV PR
    "roc_auc_test": roc_test,
    "pr_auc_test": pr_test
}]).to_csv(CSV_PATH, index=False)

# --- MLflow log (inside a run)
mlflow.set_experiment("loan_default_risk")
with mlflow.start_run(run_name="HGB_v1"):
    mlflow.log_params(hgb_rs.best_params_)
    mlflow.log_metric("roc_auc_test", roc_test)
    mlflow.log_metric("pr_auc_test", pr_test)
    mlflow.log_artifact(MODEL_PATH)
    mlflow.log_artifact(CSV_PATH)

print("Done.")
print
