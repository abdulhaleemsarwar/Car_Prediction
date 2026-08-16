# Used Car Price Predictor

An AI-powered dashboard that estimates the PKR market price of a used car,
built with **Python + Gradio** on top of a **scikit-learn `RandomForestRegressor`
Pipeline** trained on PakWheels listing data.

![Model](https://img.shields.io/badge/model-RandomForestRegressor-d64545)
![R²](https://img.shields.io/badge/R²-0.95-333)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange)

---

## How it works

```
User
 ↓
Gradio Interface (app.py)
 ↓
Input Components  (Company, Model, Year, Mileage, Engine, Specs, Registration)
 ↓
predict_price()
 ↓
Saved scikit-learn Pipeline  (model.joblib)
   ├─ ColumnTransformer
   │    ├─ StandardScaler      → numeric features
   │    └─ OneHotEncoder       → categorical features
   └─ RandomForestRegressor
 ↓
Predicted Price (PKR)
 ↓
Gradio result card
```

The Gradio app **never** recreates the `StandardScaler` / `OneHotEncoder` /
`ColumnTransformer` — all preprocessing already lives inside the saved
Pipeline. `predict_price()` just builds a one-row `pandas.DataFrame` with the
exact column names/order the Pipeline was trained on and calls
`model.predict(...)` on it.

## Project structure

```
pakwheels-price-predictor/
├── app.py                    # Gradio UI + prediction logic (run this)
├── train_model.py            # Reproducible training script (run once)
├── model.joblib               # Saved, trained Pipeline (generated)
├── metadata.json              # Dropdown choices, numeric ranges, metrics (generated)
├── Clean_Data_pakwheels.csv  # Training data
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Train the model (only needs to run once)

```bash
python train_model.py
```

This cleans the CSV the same way the original notebook did (drops the index
column, drops `Location`, drops duplicate rows), fits the Pipeline
(`StandardScaler` + `OneHotEncoder` → `RandomForestRegressor`) on a 75/25
train/test split, evaluates it, and writes two files:

- `model.joblib` — the fitted Pipeline, loaded once by the app at startup
- `metadata.json` — dropdown choices, numeric min/max/median, and the
  evaluation metrics shown in the "Model Information" panel

## Run the app

```bash
python app.py
```

Gradio will print a local URL (and, if `share=True` is added to
`demo.launch(...)`, a public one) to open in your browser.

## Features used by the model

| Type | Features |
|---|---|
| Numeric | Model Year, Mileage, Engine Capacity |
| Categorical | Company Name, Model Name, Engine Type, Color, Assembly, Body Type, Transmission Type, Registration Status |

**Note:** the dataset and the trained Pipeline only have a single
registration-related column — `Registration Status` (Registered /
Un-Registered). There is no separate "Registration Type" field in the data,
so the UI doesn't ask for one.

## Model evaluation metrics

Metrics are computed on a held-out 25% test split and loaded dynamically
from `metadata.json` (not hardcoded in the UI):

- **R²** — goodness-of-fit on the test set (not "accuracy")
- **MAE** — mean absolute error, in PKR
- **RMSE** — root-mean-square error, in PKR

## Notes

- The model is loaded once at startup (`load_model()`); the app never
  retrains on launch.
- All user inputs are validated (`validate_inputs()`) before being sent to
  the model — missing fields, out-of-range years, negative mileage, and
  non-positive engine capacity are all caught with a friendly message.
- Any unexpected error during prediction is caught and shown as a generic
  friendly message — no Python traceback is ever shown to the user (it's
  logged server-side instead).
- Picking a **Company Name** narrows the **Model Name** dropdown to that
  company's models.
