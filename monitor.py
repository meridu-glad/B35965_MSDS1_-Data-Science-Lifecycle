
# app/monitor.py
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from typing import Dict, Any, Optional
from .settings import BASELINE_STATS_PATH, FEATURE_ORDER, PSI_WARN, PSI_ALERT, KS_PVAL_WARN, ROLLING_WINDOW

# In-memory buffers (demo). In production, use a store (Blob/Azure Table/DB).
_recent_scores = []     # model probabilities
_recent_labels = []     # optional ground-truth labels for model drift

def compute_baseline_stats(df: pd.DataFrame) -> Dict[str, Any]:
    stats = {}
    for col in FEATURE_ORDER:
        series = df[col].astype(float)
        stats[col] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=1)),
            "hist_bins": list(np.histogram(series, bins=20)[1]),
            "hist_counts": list(np.histogram(series, bins=20)[0].astype(int))
        }
    return stats

def save_baseline(stats: Dict[str, Any]):
    with open(BASELINE_STATS_PATH, "w") as f:
        json.dump(stats, f)

def load_baseline() -> Optional[Dict[str, Any]]:
    try:
        with open(BASELINE_STATS_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def psi(ref_counts: np.ndarray, ref_bins: np.ndarray, cur_series: np.ndarray) -> float:
    """Population Stability Index between reference histogram and current values."""
    cur_counts, _ = np.histogram(cur_series, bins=ref_bins)
    ref_pct = (ref_counts + 1e-12) / (ref_counts.sum() + 1e-12)
    cur_pct = (cur_counts + 1e-12) / (cur_counts.sum() + 1e-12)
    diffs = cur_pct - ref_pct
    ratios = np.where(cur_pct == 0, 1e-12, cur_pct) / np.where(ref_pct == 0, 1e-12, ref_pct)
    return float(np.sum(diffs * np.log(ratios)))

def drift_report(df_current: pd.DataFrame, baseline: Dict[str, Any]) -> Dict[str, Any]:
    report = {"features": {}, "alerts": []}
    for col in FEATURE_ORDER:
        cur = df_current[col].astype(float).values
        ref = baseline[col]
        ref_bins = np.array(ref["hist_bins"], dtype=float)
        ref_counts = np.array(ref["hist_counts"], dtype=float)
        psi_score = psi(ref_counts, ref_bins, cur)
        ks_pval = ks_2samp(cur, np.repeat(ref["mean"], len(cur))).pvalue  # simple KS vs mean proxy (or store ref sample)
        status = "OK"
        if psi_score >= PSI_ALERT or ks_pval <= KS_PVAL_WARN:
            status = "ALERT"
            report["alerts"].append(col)
        elif psi_score >= PSI_WARN:
            status = "WARN"
        report["features"][col] = {
            "psi": psi_score,
            "ks_pval": float(ks_pval),
            "status": status
        }
    return report

def push_feedback(prob: float, label: Optional[int]):
    _recent_scores.append(prob)
    if label is not None:
        _recent_labels.append(label)
    # cap length
    if len(_recent_scores) > ROLLING_WINDOW:
        _recent_scores.pop(0)
    if len(_recent_labels) > ROLLING_WINDOW:
        _recent_labels.pop(0)

def model_drift() -> Dict[str, Any]:
    # Simple roll-up: if labels available, compute rolling AUC proxy (binned PR-AUC)
    if not _recent_labels or len(_recent_labels) < 50:
        return {"status": "INSUFFICIENT_LABELS", "n": len(_recent_labels)}
    # coarse PR calculation (for brevity)
    df = pd.DataFrame({"y": _recent_labels, "p": _recent_scores})
    thresholds = np.linspace(0, 1, 21)
    precisions, recalls = [], []
    for t in thresholds:
        tp = ((df["y"] == 1) & (df["p"] >= t)).sum()
        fp = ((df["y"] == 0) & (df["p"] >= t)).sum()
        fn = ((df["y"] == 1) & (df["p"] <  t)).sum()
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        precisions.append(precision); recalls.append(recall)
    pr_auc = float(np.trapz(precisions, recalls))
    status = "OK" if pr_auc >= 0.55 else ("WARN" if pr_auc >= 0.45 else "ALERT")
    return {"status": status, "rolling_pr_auc": pr_auc, "n": len(_recent_labels)}
