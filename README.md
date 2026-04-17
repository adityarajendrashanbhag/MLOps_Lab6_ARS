# Customer Churn Weak Supervision Lab

This repo is now a Streamlit app that turns the original lab idea into a customer churn workflow:

- load a bundled Telco-style churn dataset or upload your own CSV
- apply Snorkel labeling functions to generate weak labels
- inspect coverage, overlap, and conflicts
- train a lightweight churn classifier on the weak labels
- review slice-level risk patterns

## Run the app

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements_app.txt
```

3. Start Streamlit:

```bash
streamlit run streamlit_app.py
```

## App structure

- [streamlit_app.py](/d:/MLOps_Lab/MLOps_Lab6_ARS/streamlit_app.py)
- [src/data.py](/d:/MLOps_Lab/MLOps_Lab6_ARS/src/data.py)
- [src/weak_supervision.py](/d:/MLOps_Lab/MLOps_Lab6_ARS/src/weak_supervision.py)
- [src/modeling.py](/d:/MLOps_Lab/MLOps_Lab6_ARS/src/modeling.py)

## CSV expectations

The uploader works best with Telco-style columns such as:

- `tenure`
- `monthly_charges`
- `total_charges`
- `contract`
- `internet_service`
- `payment_method`
- `paperless_billing`
- `online_security`
- `tech_support`
- `senior_citizen`
- `churn`

The app also normalizes common variants like `MonthlyCharges`, `PaperlessBilling`, and `SeniorCitizen`.

## Local note

The original `requirements.txt` in this repo is locked in the current environment, so the app dependency list lives in `requirements_app.txt`.

## Why this version is stronger than the original notebook lab

- interactive instead of notebook-only
- built around customer churn, which is more portfolio-friendly
- supports custom uploads
- shows weak supervision diagnostics directly in the UI
- includes model and slice analysis in one place
