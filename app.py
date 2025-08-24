"""Streamlit dashboard for training, detecting, and explaining anomalies."""

import streamlit as st
import pandas as pd
import numpy as np
import io
import os
from src.utils import generate_synthetic_series, load_csv_timeseries
from src.model import train_neuralprophet, load_model
from src.detect import detect_anomalies, counterfactual_series, explain_anomaly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Time-Series Anomaly Detector", layout="wide")
st.title("📈 Time-Series Anomaly Detector with Counterfactual Analysis")

# Sidebar: data selection
st.sidebar.header("Data")
data_option = st.sidebar.selectbox("Data source", ["Synthetic demo", "Upload CSV"])
if data_option == "Synthetic demo":
    n = st.sidebar.number_input("Length (days)", min_value=60, max_value=3650, value=365)
    seed = st.sidebar.number_input("Random seed", value=42)
    df = generate_synthetic_series(length=n, seed=seed, spike_positions=None)
else:
    uploaded = st.sidebar.file_uploader("Upload CSV (columns: ds,y)", type=["csv"])
    if uploaded is not None:
        df = load_csv_timeseries(uploaded)
    else:
        st.sidebar.info("Upload a CSV or pick Synthetic demo.")
        st.stop()

# Sidebar: Model & detection params
st.sidebar.header("Model / Detection")
train_model_flag = st.sidebar.checkbox("Train model on this data (otherwise load saved)", value=True)
model_name = st.sidebar.text_input("Model name", value="np_model")
epochs = st.sidebar.slider("Epochs", min_value=5, max_value=500, value=60, step=5)
weekly = st.sidebar.checkbox("Weekly seasonality", value=True)
daily = st.sidebar.checkbox("Daily seasonality", value=False)
yearly = st.sidebar.checkbox("Yearly seasonality", value=False)
z_thresh = st.sidebar.slider("Anomaly z-threshold", 1.0, 6.0, 3.0, 0.1)

# Buttons
st.sidebar.header("Actions")
run_button = st.sidebar.button("Run detection")

# Main area
st.subheader("Time series (sample)")
st.write("First 10 rows:")
st.dataframe(df.head(10))

if run_button:
    # Train or load model
    with st.spinner("Training / loading model..."):
        if train_model_flag:
            model, pred = train_neuralprophet(df.copy(), epochs=epochs,
                                              weekly_seasonality=weekly,
                                              daily_seasonality=daily,
                                              yearly_seasonality=yearly,
                                              model_name=model_name)
        else:
            model = load_model(model_name)
            pred = model.predict(df.copy())

    # Detection
    df_det = detect_anomalies(df.copy(), pred, z_thresh=z_thresh)

    # Show plot with anomalies & counterfactual
    anomalies_idx = df_det.index[df_det["is_anomaly"]].tolist()
    st.write(f"Detected {len(anomalies_idx)} anomalies (z >= {z_thresh})")

    # Build counterfactual series (replace anomalies with yhat1)
    cf = counterfactual_series(df.copy(), pred, anomalies_idx)

    # Plotly: original, prediction, anomalies, counterfactual
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.02)
    fig.add_trace(go.Scatter(x=df["ds"], y=df["y"], name="Actual", mode="lines+markers"), row=1, col=1)
    if "yhat1" in pred.columns:
        fig.add_trace(go.Scatter(x=pred["ds"], y=pred["yhat1"], name="Forecast (yhat1)", mode="lines"), row=1, col=1)
    # anomalies markers
    if len(anomalies_idx) > 0:
        fig.add_trace(go.Scatter(x=df_det.loc[anomalies_idx, "ds"],
                                 y=df_det.loc[anomalies_idx, "y"],
                                 mode="markers", name="Anomalies",
                                 marker=dict(size=10, color="red", symbol="x")),
                      row=1, col=1)
    # counterfactual overlay
    fig.add_trace(go.Scatter(x=cf["ds"], y=cf["y"], name="Counterfactual (repaired)", mode="lines", line=dict(dash="dash")),
                  row=1, col=1)

    # residuals
    fig.add_trace(go.Scatter(x=df_det["ds"], y=df_det["residual"], name="Residual (y - yhat1)", mode="lines"), row=2, col=1)
    fig.add_hline(y=0, line=dict(color="black", dash="dot"))
    fig.update_layout(height=760, legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    # Table of anomalies with explanations
    st.subheader("Anomalies & explanations")
    if len(anomalies_idx) == 0:
        st.info("No anomalies detected at current threshold.")
    else:
        rows = []
        for idx in anomalies_idx:
            expl = explain_anomaly(pred, idx)
            ds = expl.get("ds")
            yhat = expl.get("yhat")
            actual = float(df_det.loc[idx, "y"])
            residual = float(df_det.loc[idx, "residual"])
            z = float(df_det.loc[idx, "residual_z"])
            comps = expl.get("components", {})
            comp_sorted = sorted(comps.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
            comp_str = ", ".join([f"{k}:{v:.3f}" for k, v in comp_sorted])
            rows.append({
                "ds": ds,
                "actual": actual,
                "predicted": yhat,
                "residual": residual,
                "z": z,
                "top_components": comp_str
            })
        st.dataframe(pd.DataFrame(rows))

    # Download repaired (counterfactual) series
    buf = io.StringIO()
    cf.to_csv(buf, index=False)
    st.download_button("Download counterfactual CSV", data=buf.getvalue(), file_name="counterfactual.csv", mime="text/csv")

    st.success("Done — explore the plots and explanations above.")
else:
    st.info("Configure options in the sidebar and click **Run detection** to train/load a model and detect anomalies.")
