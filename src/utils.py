"""Utility helpers: synthetic data generator & CSV loader."""

import numpy as np
import pandas as pd
from typing import List

def generate_synthetic_series(length: int = 365, freq: str = "D", seed: int = 42,
                              spike_positions: List[int] = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=length, freq=freq)
    t = np.arange(length).astype(float)
    trend = 0.01 * t
    weekly = 2.0 * np.sin(2 * np.pi * (t % 7) / 7.0)
    noise = rng.normal(scale=0.5, size=length)
    y = 5.0 + trend + weekly + noise
    df = pd.DataFrame({"ds": dates, "y": y})
    spike_positions = spike_positions or [int(length * 0.2), int(length * 0.6)]
    for p in spike_positions:
        if 0 <= p < length:
            df.at[p, "y"] += rng.normal(loc=8.0, scale=2.0)
    return df


def load_csv_timeseries(path: str, ds_col: str = "ds", y_col: str = "y") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={ds_col: "ds", y_col: "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    return df[["ds", "y"]]
