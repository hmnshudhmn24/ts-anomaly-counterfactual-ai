"""Detection & explainability helpers."""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from sklearn.preprocessing import StandardScaler

def _extract_components_cols(pred_df: pd.DataFrame) -> List[str]:
    comps = [c for c in pred_df.columns if c not in ("ds", "y", "yhat1", "yhat")]
    comps = [c for c in comps if np.issubdtype(pred_df[c].dtype, np.number)]
    return comps


def detect_anomalies(df: pd.DataFrame, pred: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    """Compute residuals, z-scores and flag anomalies."""
    if "yhat1" not in pred.columns and "yhat" in pred.columns:
        pred = pred.rename(columns={"yhat": "yhat1"})

    merged = pd.merge(df.reset_index(drop=True), pred[["ds", "yhat1"]], on="ds", how="left")
    merged["residual"] = merged["y"] - merged["yhat1"]
    scaler = StandardScaler()
    resid = merged["residual"].fillna(0).values.reshape(-1, 1)
    z = scaler.fit_transform(resid).flatten()
    merged["residual_z"] = z
    merged["is_anomaly"] = np.abs(merged["residual_z"]) >= z_thresh
    return merged


def counterfactual_series(df: pd.DataFrame, pred: pd.DataFrame, anomalies_idx: List[int]) -> pd.DataFrame:
    """Return a copy of df where indices in anomalies_idx are replaced by predicted values (yhat1)."""
    merged = pd.merge(df.reset_index(drop=True), pred[["ds", "yhat1"]], on="ds", how="left")
    cf = merged.copy()
    for i in anomalies_idx:
        if 0 <= i < len(cf):
            cf.at[i, "y"] = cf.at[i, "yhat1"]
    return cf[["ds", "y"]]


def explain_anomaly(pred: pd.DataFrame, idx: int) -> Dict[str, Any]:
    row = pred.iloc[idx]
    explanation = {"ds": row["ds"], "yhat": float(row.get("yhat1", np.nan))}
    comps = _extract_components_cols(pred)
    comp_values = {}
    for c in comps:
        if c in ("y", "yhat1", "yhat"):
            continue
        try:
            comp_values[c] = float(row[c])
        except Exception:
            continue
    explanation["components"] = comp_values
    return explanation
