"""
app.py
------
Used Car Price Predictor - a Gradio dashboard that serves predictions from
an already-trained scikit-learn Pipeline (StandardScaler + OneHotEncoder ->
RandomForestRegressor).

The saved Pipeline already contains all preprocessing, so this app only
ever passes raw, human-entered values into `model.predict(...)`. Nothing
here recreates encoders/scalers, and nothing here retrains the model.

Run with:
    python app.py
"""

import json
import traceback

import gradio as gr
import joblib
import pandas as pd

MODEL_PATH = "model.joblib"
METADATA_PATH = "metadata.json"


# --------------------------------------------------------------------------
# Loading (runs once, at startup)
# --------------------------------------------------------------------------
def load_model(path: str = MODEL_PATH):
    """Loads the pre-trained sklearn Pipeline. Never retrains."""
    return joblib.load(path)


def load_metadata(path: str = METADATA_PATH) -> dict:
    """Loads dropdown choices, numeric ranges, and eval metrics that were
    exported alongside the model by train_model.py."""
    with open(path, "r") as f:
        return json.load(f)


model = load_model()
metadata = load_metadata()

CHOICES = metadata["categorical_choices"]
COMPANY_TO_MODELS = metadata["company_to_models"]
RANGES = metadata["numeric_ranges"]
METRICS = metadata["metrics"]
FEATURE_ORDER = metadata["feature_order"]

ALL_MODEL_NAMES = CHOICES["Model Name"]


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate_inputs(
    company, model_name, model_year, mileage, engine_capacity,
    engine_type, color, assembly, body_type, transmission, registration_status,
):
    """Returns an error string if something is missing/out of range,
    otherwise None."""
    required = {
        "Company Name": company,
        "Model Name": model_name,
        "Engine Type": engine_type,
        "Color": color,
        "Assembly": assembly,
        "Body Type": body_type,
        "Transmission Type": transmission,
        "Registration Status": registration_status,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        return f"Please fill in: {', '.join(missing)}."

    if model_year is None:
        return "Please enter a model year."
    if mileage is None:
        return "Please enter the mileage."
    if engine_capacity is None:
        return "Please enter the engine capacity."

    yr = RANGES["Model Year"]
    if not (yr["min"] <= model_year <= 2026):
        return f"Model year should be between {yr['min']} and 2026."
    if mileage < 0:
        return "Mileage can't be negative."
    if engine_capacity <= 0:
        return "Engine capacity must be greater than 0 cc."

    return None


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def predict_price(
    company, model_name, model_year, mileage, engine_capacity,
    engine_type, color, assembly, body_type, transmission, registration_status,
):
    """Builds the single-row DataFrame the Pipeline expects (exact column
    names it was trained on) and returns the formatted result markdown."""
    error = validate_inputs(
        company, model_name, model_year, mileage, engine_capacity,
        engine_type, color, assembly, body_type, transmission, registration_status,
    )
    if error:
        return result_markdown(error=error)

    try:
        data = pd.DataFrame([{
            "Model Year": model_year,
            "Mileage": mileage,
            "Engine Capacity": engine_capacity,
            "Company Name": company,
            "Model Name": model_name,
            "Engine Type": engine_type,
            "Color": color,
            "Assembly": assembly,
            "Body Type": body_type,
            "Transmission Type": transmission,
            "Registration Status": registration_status,
        }])[FEATURE_ORDER]

        prediction = model.predict(data)[0]
        return result_markdown(price=prediction)

    except Exception:
        # Never leak a traceback to the user.
        print(traceback.format_exc())
        return result_markdown(
            error="Unable to generate the prediction. Please check your inputs and try again."
        )


def result_markdown(price: float | None = None, error: str | None = None) -> str:
    if error:
        return f"""
<div class="result-box result-error">
    <div class="result-label">Couldn't estimate price</div>
    <div class="result-error-text">{error}</div>
</div>
"""
    if price is not None:
        formatted = f"PKR {price:,.0f}"
        return f"""
<div class="result-box result-ready">
    <div class="result-label">Estimated Market Price</div>
    <div class="result-price">{formatted}</div>
    <div class="result-note">Based on the vehicle details you provided</div>
</div>
"""
    return """
<div class="result-box result-empty">
    <div class="result-label">Estimated Market Price</div>
    <div class="result-placeholder">Enter your vehicle information<br>and click Predict Price.</div>
</div>
"""


# --------------------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------------------
def update_model_choices(company: str):
    """Cascading dropdown: narrows Model Name choices to the selected
    Company Name."""
    models = COMPANY_TO_MODELS.get(company, ALL_MODEL_NAMES)
    return gr.update(choices=models, value=None)


def reset_form():
    return (
        None, None, RANGES["Model Year"]["median"], RANGES["Mileage"]["median"],
        RANGES["Engine Capacity"]["median"], None, None, None, None, None, None,
        result_markdown(),
    )


CUSTOM_CSS = """
:root {
    --accent: #d64545;
    --accent-dark: #a83232;
    --ink: #17191c;
    --paper: #faf9f7;
}
.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}
#hero {
    text-align: center;
    padding: 36px 20px 20px 20px;
    background: linear-gradient(135deg, #17191c 0%, #2b2f36 60%, #3a3024 100%);
    border-radius: 18px;
    margin-bottom: 22px;
    color: #f5f2ec;
}
#hero h1 {
    font-size: 2.3rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0 0 8px 0;
    color: #ffffff;
}
#hero p {
    font-size: 1.05rem;
    color: #cfd2d8;
    margin: 0;
}
.section-card {
    background: var(--paper);
    border: 1px solid #e7e3db;
    border-radius: 14px;
    padding: 18px 20px 6px 20px;
    margin-bottom: 16px;
}
.section-title {
    font-weight: 700;
    font-size: 1.05rem;
    color: var(--ink);
    margin-bottom: 4px;
    border-left: 4px solid var(--accent);
    padding-left: 10px;
}
.result-box {
    border-radius: 16px;
    padding: 26px 22px;
    text-align: center;
    margin-top: 6px;
}
.result-empty { background: #f1efe9; border: 1px dashed #c9c3b6; }
.result-ready {
    background: linear-gradient(135deg, #1f2226 0%, #33210f 100%);
    color: #fff;
}
.result-error { background: #fdecea; border: 1px solid #f3b3ac; }
.result-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    opacity: 0.75;
    margin-bottom: 8px;
}
.result-price {
    font-size: 2.4rem;
    font-weight: 800;
    color: #ffbf69;
}
.result-note { font-size: 0.85rem; opacity: 0.7; margin-top: 6px; }
.result-placeholder { font-size: 1rem; color: #6b665c; line-height: 1.5; }
.result-error-text { color: #a12f24; font-weight: 600; }
.steps-row { text-align: center; }
.step-num {
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--accent);
}
.step-title { font-weight: 700; margin: 4px 0; }
.step-desc { font-size: 0.9rem; color: #5a5750; }
.metric-box {
    text-align: center;
    background: var(--paper);
    border: 1px solid #e7e3db;
    border-radius: 12px;
    padding: 14px 8px;
}
.metric-value { font-size: 1.4rem; font-weight: 800; color: var(--ink); }
.metric-label { font-size: 0.78rem; color: #77726a; text-transform: uppercase; letter-spacing: 0.5px; }
"""


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------
def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Used Car Price Predictor") as demo:

        gr.HTML(
            """
            <div id="hero">
                <h1>🚗 Used Car Price Predictor</h1>
                <p>Get an AI-powered estimate of your car's market value.</p>
            </div>
            """
        )

        with gr.Row():
            # ---------------- Left column: the form ----------------
            with gr.Column(scale=3):
                with gr.Group(elem_classes="section-card"):
                    gr.Markdown("### Vehicle Information", elem_classes="section-title")
                    with gr.Row():
                        company = gr.Dropdown(
                            choices=CHOICES["Company Name"], label="Company Name",
                            info="e.g. Toyota, Suzuki, Honda", interactive=True,
                        )
                        model_name = gr.Dropdown(
                            choices=ALL_MODEL_NAMES, label="Model Name",
                            info="Pick a company first to narrow this list", interactive=True,
                        )
                    with gr.Row():
                        model_year = gr.Number(
                            label="Model Year", value=RANGES["Model Year"]["median"],
                            minimum=RANGES["Model Year"]["min"], maximum=2026, precision=0,
                            info=f"{RANGES['Model Year']['min']}–2026",
                        )
                        mileage = gr.Number(
                            label="Mileage (km)", value=RANGES["Mileage"]["median"],
                            minimum=0, info="Total kilometers driven",
                        )
                        engine_capacity = gr.Number(
                            label="Engine Capacity (cc)", value=RANGES["Engine Capacity"]["median"],
                            minimum=1, info="e.g. 1000, 1300, 1800",
                        )

                with gr.Group(elem_classes="section-card"):
                    gr.Markdown("### Specifications", elem_classes="section-title")
                    with gr.Row():
                        engine_type = gr.Dropdown(choices=CHOICES["Engine Type"], label="Engine Type")
                        body_type = gr.Dropdown(choices=CHOICES["Body Type"], label="Body Type")
                    with gr.Row():
                        transmission = gr.Dropdown(choices=CHOICES["Transmission Type"], label="Transmission Type")
                        assembly = gr.Dropdown(choices=CHOICES["Assembly"], label="Assembly")
                        color = gr.Dropdown(choices=CHOICES["Color"], label="Color")

                with gr.Group(elem_classes="section-card"):
                    gr.Markdown("### Registration", elem_classes="section-title")
                    registration_status = gr.Dropdown(
                        choices=CHOICES["Registration Status"], label="Registration Status",
                        info="Note: the trained model does not use a separate 'Registration Type' field — only Registration Status.",
                    )

                with gr.Row():
                    clear_btn = gr.Button("Reset", variant="secondary")
                    predict_btn = gr.Button("Predict Price", variant="primary", size="lg")

            # ---------------- Right column: result + info ----------------
            with gr.Column(scale=2):
                result_display = gr.HTML(result_markdown())

                with gr.Group(elem_classes="section-card"):
                    gr.Markdown("### Model Information", elem_classes="section-title")
                    gr.Markdown("**Random Forest Regressor**")
                    with gr.Row():
                        gr.HTML(f"""<div class="metric-box"><div class="metric-value">{METRICS['r2']*100:.1f}%</div><div class="metric-label">R²</div></div>""")
                        gr.HTML(f"""<div class="metric-box"><div class="metric-value">{METRICS['mae']:,.0f}</div><div class="metric-label">MAE (PKR)</div></div>""")
                        gr.HTML(f"""<div class="metric-box"><div class="metric-value">{METRICS['rmse']:,.0f}</div><div class="metric-label">RMSE (PKR)</div></div>""")
                    gr.Markdown(
                        "_R² measures how well the model's predictions fit the "
                        "evaluation data — it is a goodness-of-fit statistic, "
                        "not a percentage of correct predictions. MAE and RMSE "
                        "are the average and root-mean-square prediction error, "
                        "in PKR, on held-out test data._"
                    )

                with gr.Group(elem_classes="section-card"):
                    gr.Markdown("### How It Works", elem_classes="section-title")
                    with gr.Row(elem_classes="steps-row"):
                        gr.HTML('<div class="step-num">01</div><div class="step-title">Enter Details</div><div class="step-desc">Provide the vehicle specifications.</div>')
                        gr.HTML('<div class="step-num">02</div><div class="step-title">AI Analysis</div><div class="step-desc">The trained model analyzes the vehicle.</div>')
                        gr.HTML('<div class="step-num">03</div><div class="step-title">Get Estimate</div><div class="step-desc">Receive the predicted market price in PKR.</div>')

        inputs = [
            company, model_name, model_year, mileage, engine_capacity,
            engine_type, color, assembly, body_type, transmission, registration_status,
        ]

        company.change(fn=update_model_choices, inputs=company, outputs=model_name)

        predict_btn.click(
            fn=predict_price, inputs=inputs, outputs=result_display,
            show_progress="full",
        )

        clear_btn.click(
            fn=reset_form, inputs=None,
            outputs=inputs + [result_display],
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    # theme/css are passed here (Gradio 6.x moved them out of the Blocks
    # constructor); older Gradio (<6) accepts these same kwargs on launch too.
    demo.launch(css=CUSTOM_CSS, theme=gr.themes.Soft(primary_hue="red"))
