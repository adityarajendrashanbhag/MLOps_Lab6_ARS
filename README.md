# Streamlit Lab: Customer Churn Predictor

This repository contains a simple Streamlit lab project for customer churn prediction.

The app takes telecom customer details as input and predicts whether the customer is likely to churn. The goal is to demonstrate a clean Streamlit interface with one-click prediction.

## UI Screenshot

Add one screenshot of the app here before submission.

![UI Screenshot](./assets/ui-screenshot-placeholder.png)

## What The App Does

- collects relevant customer inputs
- predicts churn with one click
- shows churn probability
- assigns a simple risk band: `Low`, `Medium`, or `High`

## How It Works

- the app is implemented in [streamlit_app.py](/d:/MLOps_Lab/MLOps_Lab6_ARS/streamlit_app.py:1)
- a synthetic telecom churn dataset is generated in memory
- a `LogisticRegression` model is trained using Scikit-learn
- user inputs are converted into a one-row DataFrame for prediction

Synthetic data is used so the project runs without any external dataset or backend.

## Input Features

- tenure
- monthly charges
- total charges
- contract
- internet service
- payment method
- paperless billing
- senior citizen
- partner
- dependents
- online security
- tech support

## Files

- [streamlit_app.py](/d:/MLOps_Lab/MLOps_Lab6_ARS/streamlit_app.py:1): main Streamlit app
- [tests/test_churn.py](/d:/MLOps_Lab/MLOps_Lab6_ARS/tests/test_churn.py:1): tests for prediction logic
- [requirements_app.txt](/d:/MLOps_Lab/MLOps_Lab6_ARS/requirements_app.txt:1): dependencies

## Run The App

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements_app.txt
python -m streamlit run streamlit_app.py
```

## Run Tests

```powershell
python -m pytest tests/test_churn.py -q
```

## Notes

- this is a Streamlit lab submission, so the focus is on UI and prediction flow
- the model is trained on synthetic data for demonstration purposes
