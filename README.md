# Customer Churn Predictor

This repository contains a simple Streamlit application for predicting whether a telecom customer is likely to churn.

The project is intentionally small and focused on the Streamlit lab requirement: collect user inputs in a web interface, run a model, and display the result clearly.

## Project Objective

The app allows a user to:

- enter telecom customer information
- click one button
- receive an immediate churn prediction
- view the churn probability and a simple risk band

This is a self-contained demo application. It does not require an external dataset, backend API, or database.

## How The App Works

The application is implemented in [streamlit_app.py](/d:/MLOps_Lab/MLOps_Lab6_ARS/streamlit_app.py:1).

When the app starts:

1. A synthetic telecom churn dataset is generated in memory.
2. A `LogisticRegression` model is trained on that dataset.
3. The trained model is cached with Streamlit so it is not retrained on every interaction.

When the user submits the form:

1. The input fields are converted into a one-row pandas DataFrame.
2. The model predicts churn probability.
3. The app displays:
   - predicted outcome
   - churn probability
   - risk band: `Low`, `Medium`, or `High`
   - submitted customer details

## Input Features

The app uses the following customer attributes:

- `tenure`
- `monthly_charges`
- `total_charges`
- `contract`
- `internet_service`
- `payment_method`
- `paperless_billing`
- `senior_citizen`
- `partner`
- `dependents`
- `online_security`
- `tech_support`

These features are stored in the `FEATURE_COLUMNS` list in [streamlit_app.py](/d:/MLOps_Lab/MLOps_Lab6_ARS/streamlit_app.py:56).

## Model Details

The app uses a Scikit-learn pipeline with:

- `StandardScaler` for numeric features
- `OneHotEncoder(handle_unknown="ignore")` for categorical features
- `LogisticRegression` as the classifier

This design keeps the project easy to explain and suitable for a Streamlit classroom demo.

## Why Synthetic Data Is Used

Synthetic data is used so the project can run immediately without requiring a CSV file or a separate training step.

The synthetic training data is generated using business-style churn assumptions such as:

- month-to-month contracts having higher churn risk
- fiber optic service having higher churn risk
- electronic check payments having higher churn risk
- longer tenure reducing churn risk
- online security and tech support reducing churn risk

This makes the demo realistic enough for a lab while keeping the repository small and reproducible.

## User Interface

The Streamlit UI includes:

- a title and short description
- a prediction form with relevant telecom inputs
- a `Predict Churn` button
- a result section showing prediction and probability
- a small summary table of submitted values

Custom CSS is used in [streamlit_app.py](/d:/MLOps_Lab/MLOps_Lab6_ARS/streamlit_app.py:14) to improve readability and make the app look cleaner.

## Repository Structure

Current project files:

- [streamlit_app.py](/d:/MLOps_Lab/MLOps_Lab6_ARS/streamlit_app.py:1)
  Main Streamlit application and model logic.

- [tests/test_churn.py](/d:/MLOps_Lab/MLOps_Lab6_ARS/tests/test_churn.py:1)
  Automated tests for model behavior and helper logic.

- [tests/__init__.py](/d:/MLOps_Lab/MLOps_Lab6_ARS/tests/__init__.py:1)
  Test package marker.

- [requirements_app.txt](/d:/MLOps_Lab/MLOps_Lab6_ARS/requirements_app.txt:1)
  Python dependencies required to run the app.

- [.gitignore](/d:/MLOps_Lab/MLOps_Lab6_ARS/.gitignore:1)
  Ignores local environments, caches, editor files, and other non-source files.

## Testing

The repository includes a test suite in [tests/test_churn.py](/d:/MLOps_Lab/MLOps_Lab6_ARS/tests/test_churn.py:1).

The tests cover:

- customer input DataFrame construction
- model pipeline structure
- prediction output format
- probability range validation
- high-risk and low-risk profile behavior
- risk band logic
- summary table formatting
- edge cases such as unknown categories

This helps verify that the prediction flow is stable and that the app logic matches the intended behavior.

## Installation

Create and activate a virtual environment, then install the dependencies.

### Windows

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
py -3.12 -m pip install -r requirements_app.txt
```

### Alternative

If `py -3.12` is not available:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements_app.txt
```

## Running The App

Start Streamlit from the project root:

```powershell
python -m streamlit run streamlit_app.py
```

Streamlit will print a local URL, usually:

```text
http://localhost:8501
```

## Running Tests

To run the automated tests:

```powershell
python -m pytest tests/test_churn.py -q
```

If `pytest` is not installed in your environment, install it with:

```powershell
python -m pip install pytest
```

## Expected Output

After submitting the form, the app displays:

- whether the customer is likely to churn or stay
- the churn probability as a percentage
- the risk band
- the submitted customer values

## Notes For Evaluation

This project is designed as a Streamlit lab submission, so the emphasis is on:

- interactive UI design
- one-click prediction
- clean organization
- understandable modeling logic
- basic testing

It is not intended to be a production MLOps system. The model is trained on synthetic data for demonstration purposes only.

## Summary

In short, this repository demonstrates a complete but lightweight Streamlit machine learning app:

- a user-friendly prediction form
- a simple churn model
- immediate prediction output
- automated tests for the main logic

This makes it suitable for teaching, demonstration, and lab evaluation.
