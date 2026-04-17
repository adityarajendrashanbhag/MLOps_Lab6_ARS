# Customer Churn Weak Supervision Lab

This repository is a Streamlit application that demonstrates how to use weak supervision for customer churn prediction.

Instead of hand-labeling every row, the app uses a small set of heuristic rules, called labeling functions, to generate noisy labels. Those weak labels are then combined with Snorkel's `LabelModel` to produce probabilistic churn signals, which are used to train a simple downstream classifier.

The project was rebuilt from an earlier lab reference into a customer churn use case that is more practical, more portfolio-friendly, and easier to demo interactively.

## What This App Does

The app supports two modes:

1. Use a bundled demo dataset that mimics a Telco churn dataset.
2. Upload your own churn CSV and run the same pipeline on it.

Inside the app you can:

- inspect the dataset
- apply heuristic churn rules
- review labeling function coverage, overlap, and conflicts
- generate weak labels and churn probabilities
- train a lightweight churn model on those weak labels
- inspect slice-level behavior for important customer groups

## Core Idea

Traditional supervised learning needs labeled training data. In many real projects, that labeling process is expensive and slow.

Weak supervision is an alternative approach:

- you write rules based on domain knowledge
- each rule votes `churn`, `retain`, or `abstain`
- rules may disagree or be imperfect
- Snorkel learns how to combine those noisy signals

This is especially useful when:

- a dataset is large
- labels are incomplete or expensive
- business heuristics already exist
- you want a first working system before a full annotation effort

## End-to-End Pipeline

The app follows this flow:

1. Load data
2. Standardize column names and common Telco field variants
3. Apply labeling functions
4. Build a label matrix
5. Estimate weak labels with Snorkel
6. Attach probabilistic churn scores to each row
7. Train a logistic regression model on weak labels
8. Evaluate against the gold churn column if the uploaded dataset has one
9. Show slice-level summaries for key customer groups

## Labeling Functions Included

The current rules are implemented in [src/weak_supervision.py](/d:/MLOps_Lab/MLOps_Lab6_ARS/src/weak_supervision.py:31).

They encode simple business logic:

- `lf_month_to_month_fiber_risk`
  Flags customers on expensive month-to-month fiber plans as churn risk.
- `lf_new_customer_echeck_risk`
  Flags short-tenure customers using electronic checks and paperless billing.
- `lf_senior_high_bill_risk`
  Flags senior customers with very high monthly charges.
- `lf_long_term_secure_customer`
  Marks long-tenure customers with longer contracts and tech support as stable.
- `lf_protection_bundle_safe`
  Marks customers with online security and device protection as lower risk.
- `lf_low_charge_no_internet_safe`
  Marks low-charge customers without internet service as likely retain.

These rules are intentionally simple. They are meant to be understandable and easy to extend, not perfect.

## Project Structure

```text
MLOps_Lab6_ARS/
|-- streamlit_app.py
|-- requirements_app.txt
|-- README.md
|-- src/
|   |-- __init__.py
|   |-- data.py
|   |-- weak_supervision.py
|   |-- modeling.py
|-- assets/
```

### Main files

- [streamlit_app.py](/d:/MLOps_Lab/MLOps_Lab6_ARS/streamlit_app.py:1)
  Streamlit UI, app layout, charts, tabs, and orchestration.

- [src/data.py](/d:/MLOps_Lab/MLOps_Lab6_ARS/src/data.py:13)
  Demo data generation, column normalization, and upload parsing.

- [src/weak_supervision.py](/d:/MLOps_Lab/MLOps_Lab6_ARS/src/weak_supervision.py:96)
  Labeling functions, Snorkel application, label matrix analysis, and weak label generation.

- [src/modeling.py](/d:/MLOps_Lab/MLOps_Lab6_ARS/src/modeling.py:19)
  Downstream logistic regression model, feature preprocessing, metrics, and slice summaries.

## Streamlit App Sections

The interface is organized into tabs:

### Overview

- preview the dataset
- review weak-label distribution
- inspect observed churn rate if a gold target exists

### Rules

- see the current rulebook
- inspect labeling function statistics such as coverage and behavior

### Weak Labels

- review row-level churn probabilities
- inspect the first rows of the label matrix
- visualize the churn probability distribution

### Model

- train a lightweight classifier
- inspect holdout metrics
- review feature importance

### Slices

- compare model behavior for important customer segments
- review churn scores by contract type and other business slices

## Dataset Expectations

The app works best with Telco-style customer churn data.

Expected or supported columns include:

- `customer_id`
- `tenure`
- `monthly_charges`
- `total_charges`
- `contract`
- `internet_service`
- `payment_method`
- `paperless_billing`
- `online_security`
- `device_protection`
- `tech_support`
- `senior_citizen`
- `churn`

The upload pipeline also normalizes common variants such as:

- `MonthlyCharges` -> `monthly_charges`
- `TotalCharges` -> `total_charges`
- `InternetService` -> `internet_service`
- `PaperlessBilling` -> `paperless_billing`
- `SeniorCitizen` -> `senior_citizen`

If a target column like `churn`, `attrition`, `exited`, `target`, or `label` is present, the app will use it for evaluation.

## Demo Dataset

The bundled demo dataset is generated in [src/data.py](/d:/MLOps_Lab/MLOps_Lab6_ARS/src/data.py:141).

It simulates realistic churn behavior by combining:

- contract type
- internet service type
- payment method
- tenure
- monthly charges
- support and protection services

This makes the app runnable even if you do not yet have a real churn CSV ready.

## Installation

Create a virtual environment and install the app dependencies.

### Windows

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
py -3.12 -m pip install -r requirements_app.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements_app.txt
```

## Running the App

Start Streamlit with:

```bash
streamlit run streamlit_app.py
```

Then open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## How To Use The App

### Option 1: Use the built-in demo dataset

1. Launch the app.
2. Keep `Dataset source` set to `Demo dataset`.
3. Choose demo row count and random seed from the sidebar.
4. Review the generated weak labels and model output.

### Option 2: Upload your own churn CSV

1. Launch the app.
2. Change `Dataset source` to `Upload CSV`.
3. Upload a churn dataset in CSV format.
4. Check whether the target column was detected.
5. Explore the rules, weak labels, and model results.

## Modeling Details

The downstream model in [src/modeling.py](/d:/MLOps_Lab/MLOps_Lab6_ARS/src/modeling.py:96) is a logistic regression classifier.

The preprocessing pipeline:

- imputes missing numeric values with the median
- scales numeric features
- imputes missing categorical values with the most frequent category
- one-hot encodes categorical features

Why logistic regression was chosen:

- fast to train
- easy to explain
- a good baseline for tabular churn data
- appropriate for a weak-supervision teaching app

## Metrics Shown

When a valid gold target exists, the app reports:

- accuracy
- precision
- recall
- F1 score
- ROC AUC

These metrics evaluate the downstream classifier against the dataset's actual churn column, not just the weak labels.

## Slice Analysis

The app currently checks a few important business slices in [src/modeling.py](/d:/MLOps_Lab/MLOps_Lab6_ARS/src/modeling.py:72):

- month-to-month customers
- fiber optic customers
- senior citizens
- customers with high monthly charges

This helps answer questions like:

- Is the model treating high-risk groups differently?
- Are churn probabilities inflated in one segment?
- Which slices deserve better rules?

## Why This Project Is Useful

This project is a good reference for:

- weak supervision with Snorkel
- interactive ML demos in Streamlit
- translating notebook experiments into an app
- building an explainable churn prototype
- combining heuristic labeling with lightweight modeling

It is also a stronger portfolio project than a basic tutorial clone because it turns a general lab concept into a practical churn use case.

## Extending The App

Good next improvements would be:

- add a rule editor in the UI
- allow users to toggle labeling functions on and off
- add downloadable scored CSV output
- add confusion matrix and precision-recall curve plots
- support user-defined slice filters
- compare majority vote vs `LabelModel` directly in the app
- add tests for data normalization and labeling behavior

## Known Local Environment Notes

During development in this workspace, there were two environment-specific issues:

- the original `requirements.txt` file was locked and could not be replaced cleanly
- the local Python launcher was pointing at a broken Windows Store shim

Because of that, the active dependency file for this app is:

- [requirements_app.txt](/d:/MLOps_Lab/MLOps_Lab6_ARS/requirements_app.txt:1)

If your local environment is healthy, using `requirements_app.txt` is enough to run the project.

## Summary

This repository now contains a full customer churn weak supervision demo:

- Streamlit front end
- bundled demo data
- Snorkel-based rule aggregation
- downstream churn model
- slice analysis

If you want to continue building on top of it, the best next step is to make the rules editable from the Streamlit sidebar and let users export scored results.
