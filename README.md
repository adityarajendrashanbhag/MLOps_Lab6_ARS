# Customer Churn Predictor

This repository is now a minimal Streamlit lab project.

The app takes telecom customer details as input and predicts whether the customer is likely to churn. The goal is very simple:

- enter relevant customer information
- click `Predict Churn`
- get the result immediately

## Project files

- [streamlit_app.py](/d:/MLOps_Lab/MLOps_Lab6_ARS/streamlit_app.py:1)
- [requirements_app.txt](/d:/MLOps_Lab/MLOps_Lab6_ARS/requirements_app.txt:1)
- [.gitignore](/d:/MLOps_Lab/MLOps_Lab6_ARS/.gitignore:1)

## Inputs used

The app uses these fields:

- tenure
- monthly charges
- total charges
- contract type
- internet service
- payment method
- paperless billing
- senior citizen
- partner
- dependents
- online security
- tech support

## How it works

The app trains a lightweight logistic regression model on realistic synthetic telecom churn data when the app starts. That keeps the project self-contained and easy to run for a Streamlit lab.

## Run the app

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
py -3.12 -m pip install -r requirements_app.txt
py -3.12 -m streamlit run streamlit_app.py
```

If `py -3.12` does not work, use:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements_app.txt
python -m streamlit run streamlit_app.py
```

## Output

After clicking the button, the app shows:

- prediction result
- churn probability
- risk band

This version is intentionally small and focused on the Streamlit interaction itself.
