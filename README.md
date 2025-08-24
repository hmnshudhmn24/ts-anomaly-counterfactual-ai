# 🕵️‍♂️ Time-Series Anomaly Detector with Counterfactual Analysis

An explainable time-series monitoring dashboard using **NeuralProphet** for forecasting and a residual-based detector for anomalies. The app lets you:

- Train or load a NeuralProphet model 📚  
- Detect anomalous points via residual z-score (configurable threshold) 🚨  
- Produce a **counterfactual (repaired)** series where anomalies are replaced with model forecasts 🔁  
- Explain anomalies by showing model components (trend, seasonalities) and top contributing components 🧭  
- Interactively explore data via a **Streamlit** dashboard 📊

---

## Quickstart 🚀

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run the Streamlit dashboard
streamlit run app.py
```

### Files included
```
ts-anomaly-counterfactual-ai/
├─ app.py                 # Streamlit dashboard (UI)
├─ requirements.txt
├─ src/
│  ├─ model.py            # NeuralProphet training, save/load helpers
│  ├─ detect.py           # anomaly detection, counterfactual creation, explanations
│  └─ utils.py            # synthetic dataset generator & helper IO
├─ demo_data.csv          # small synthetic demo dataset (optional)
└─ README.md
```

## How it works (short)
1. Train NeuralProphet on your series (one-step forecasting).  
2. Compute residuals `y - yhat1` and z-score them.  
3. Flag anomalies where `|z| >= threshold`.  
4. Create a counterfactual series by replacing anomalies with `yhat1`.  
5. Explain anomalies by reporting top model component contributions and residual magnitude.

---

## Notes & Extensions
- If NeuralProphet is hard to install on your platform, try running on Colab or use an alternative (Prophet, PyCaret, or a simple ARIMA).  
- For production, consider streaming ingestion, robust detectors (MAD/IsolationForest), and automatic alerting.  

---

## License
MIT — adapt and extend! ⭐
