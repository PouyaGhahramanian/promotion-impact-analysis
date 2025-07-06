# Promotion Impact Analysis

This repository analyzes the impact of in-store promotions using historical sales data. It provides both a **naive cluster-based forecasting baseline** and a **machine learning-based forecasting pipeline** using LightGBM, XGBoost, or CatBoost.

## 📊 Project Structure

```
promotion-impact-analysis/
├── data/                 # Input CSV files
├── figures/              # Saved plots and figures
├── models/               # Trained ML models
├── results/              # Forecast outputs
├── src/                  # All source code
│   ├── config.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── forecaster.py
│   ├── model.py
│   └── visualizer.py
├── main.py               # Main entry point
├── requirements.txt
└── .gitignore
```

## 🚀 How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the pipeline**
   ```bash
   python main.py
   ```

3. **Outputs**
   - Visualizations: `figures/`
   - Forecasts: `results/`
   - Trained models: `models/`

## 🧠 Features

- **Sales data preprocessing**
- **Promotion tagging** (Promo1–4 for training, Promo5 for test)
- **Cluster-based lift modeling** (Slow/Medium/Fast for Items and Stores)
- **Feature engineering** (rolling averages, weekend flag, promo lags, etc.)
- **Promotion 5 forecasting** using:
  - A naive method (baseline + lift)
  - A supervised regression model (LightGBM/XGBoost/CatBoost)
- **Evaluation metrics**
  - MAE, RMSE, NRMSE
- **Visualization**
  - Sales trends, cluster distributions, return rates, lift plots

## 📈 Model Options

Set your desired model in `config.py`:

```python
MODEL_NAME = 'LGBM'  # Options: 'LGBM', 'XGB', 'CAT'
```

## ⚙️ Configuration

Edit `src/config.py` to control:

- Feature engineering
- Model type (static/stream)
- Features to include
- Sample size

## 📝 Notes

- Negative quantities represent returns.
- If some clusters have no lift value, zero lift is assumed.

## 📄 License

MIT License

