# Bankruptcy Early Warning Indicator

Streamlit app that scores corporate bankruptcy / financial-distress risk from company financials, with SHAP explanations, scenario simulation, portfolio batch scoring, and PDF export.

## Problem

Credit and risk teams need an early signal of distress before it shows up in rating actions. Raw financial statements are hard to compare across companies; this tool turns them into a calibrated probability plus the drivers behind it.

## Method

- **Model:** CatBoost classifier (`model/model.pkl`) with calibrated probabilities
- **Features:** engineered ratios (leverage, margins, working capital, revenue efficiency, and related fields) via `utils/feature_engineering.py`
- **Explainability:** SHAP local explanations for every single-company score
- **Workflows:**
  - Single-company analysis
  - Scenario simulator (revenue / cost / leverage shocks)
  - Portfolio batch scoring (CSV / XLSX upload)
  - Reports and PDF export
  - Optional AI narrative summary when an Anthropic API key is set

## Results / what you get

- Distress probability and risk label for a company or portfolio
- Top SHAP drivers and industry benchmark comparison (CSV-backed)
- Downloadable PDF deliverables from the Reports page

This repo does not include a live hosted demo. Run locally to explore.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Optional: set `ANTHROPIC_API_KEY` in the environment if you want AI-assisted report language (`utils/ai_summary.py`).

## Project layout

```
app.py                 # Streamlit entry / overview
pages/                 # Analysis, Simulator, Portfolio, Reports
utils/                 # model load, features, SHAP, plots, PDF
model/model.pkl        # trained CatBoost model
data/industry_benchmarks.csv
assets/logo.png
```

## Stack

`Python` · `Streamlit` · `CatBoost` · `SHAP` · `Plotly` · `pandas` · `ReportLab`
