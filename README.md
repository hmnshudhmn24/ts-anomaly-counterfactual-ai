
# 📈🚨 Time-Series Anomaly Detector with Counterfactual Analysis

An interactive **time-series anomaly detection dashboard** powered by **NeuralProphet**.  
This tool not only detects anomalies but also **explains why** a point is anomalous using **counterfactual comparisons** to expected (normal) behavior.  

🔮 Perfect for showcasing skills in **time-series forecasting, anomaly detection, explainable AI, and dashboard development**!



## ✨ Features

- 📊 **Time-Series Forecasting** with NeuralProphet
- 🚨 **Automatic anomaly detection** using deviation from forecasted values
- 🔍 **Counterfactual explanations** → compare anomalous points with their expected “normal” counterparts
- 🖥 **Streamlit dashboard** for interactive data exploration and visualization
- 🧪 Includes **synthetic demo dataset** (energy-style data with injected anomalies)
- 🛠 Modular code with `src/` for easy extension and training on real datasets



## 📂 Project Structure

```
ts-anomaly-counterfactual-ai/
├─ app.py                 # Streamlit dashboard (UI)
├─ requirements.txt       # dependencies
├─ src/
│  ├─ model.py            # NeuralProphet training, save/load helpers
│  ├─ detect.py           # anomaly detection + counterfactual creation
│  └─ utils.py            # synthetic dataset generator & IO helpers
├─ demo_data.csv          # small synthetic demo dataset
└─ README.md              # project documentation
```



## 🚀 Getting Started

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Train or load model
```bash
python -m src.model --train --data demo_data.csv --save models/demo_model.pkl
```

### 3️⃣ Run anomaly detection
```bash
python -m src.detect --data demo_data.csv --model models/demo_model.pkl
```

### 4️⃣ Launch Streamlit dashboard
```bash
streamlit run app.py
```



## 🖼 Dashboard Preview

The dashboard provides:  
- 📈 Raw data vs Forecast  
- 🚨 Highlighted anomalies  
- 🔍 Counterfactual values (expected “normal” behavior for anomaly points)  

Example visualization:  

✅ Normal data → follows forecast  
⚠️ Anomaly → flagged with red markers, counterfactual values shown as dashed lines



## 🧩 How It Works

1. **Model Training**  
   - NeuralProphet is trained on time-series data  
   - Captures seasonality + trend  

2. **Forecasting**  
   - Predicts expected values for each timestamp  

3. **Anomaly Detection**  
   - If `|observed - forecast| > threshold`, flag as anomaly  

4. **Counterfactual Explanation**  
   - Replaces anomaly with forecasted value  
   - Displays side-by-side: “observed vs expected”



## 📊 Example (Synthetic Energy Data)

| Date       | Observed | Forecast | Anomaly | Counterfactual |
|------------|----------|----------|---------|----------------|
| 2023-05-10 | 420      | 310      | ✅ Yes  | 310            |
| 2023-05-11 | 305      | 298      | ❌ No   | -              |
| 2023-05-12 | 480      | 315      | ✅ Yes  | 315            |



## 🛠 Tech Stack

- 🧠 **NeuralProphet** → forecasting backbone
- 🔎 **Counterfactual analysis** → explainability
- 📊 **Pandas + Matplotlib/Plotly** → data visualization
- 🎛 **Streamlit** → interactive dashboard
- 🐍 **Python** (3.8+)



## 📌 Use Cases

- ⚡ **Energy usage monitoring** (detect unusual spikes in consumption)
- 🏭 **Industrial IoT** (monitor sensor readings for faults)
- 💰 **Finance** (flag suspicious trading volumes or stock anomalies)
- 🏥 **Healthcare** (detect abnormal patient vitals)



## 🧪 Next Steps

- ✅ Add **real-world datasets** (energy, stock market, IoT sensors)
- ✅ Improve counterfactuals with **autoencoder-based reconstructions**
- 🔜 Deploy dashboard as **web app (Heroku/Streamlit Cloud)**
- 🔜 Integrate with **alerts/notifications** (email/Slack)


