"""Model helpers: train, save, load using NeuralProphet."""

from neuralprophet import NeuralProphet
import pandas as pd
import os
from typing import Tuple

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def train_neuralprophet(df: pd.DataFrame, 
                        epochs: int = 50,
                        weekly_seasonality: bool = True,
                        yearly_seasonality: bool = False,
                        daily_seasonality: bool = False,
                        n_lags: int = 0,
                        model_name: str = "np_model") -> Tuple[NeuralProphet, pd.DataFrame]:
    """
    Train a NeuralProphet model on dataframe `df` which must have columns:
      - 'ds' (datetime-like)
      - 'y'  (target)
    Returns (model, forecast_df) where forecast_df is model.predict(df).
    """
    m = NeuralProphet(
        n_forecasts=1,
        n_lags=n_lags,
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
    )

    # Fit
    m.fit(df, freq="D", epochs=epochs, progress="none")
    # Predict on training set (one-step forecasts)
    forecast = m.predict(df)
    # Save model
    fname = os.path.join(MODEL_DIR, f"{model_name}.pth")
    m.save(fname)
    return m, forecast


def load_model(model_name: str = "np_model") -> NeuralProphet:
    path = os.path.join(MODEL_DIR, f"{model_name}.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}. Train a model first.")
    m = NeuralProphet.load(path)
    return m
