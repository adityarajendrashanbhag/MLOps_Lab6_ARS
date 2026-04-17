"""
Tests for the Telecom Customer Churn Predictor.

Covers:
- Customer frame construction
- Model training and prediction
- Probability range validation
- Risk band logic
- High-risk vs low-risk customer profiles
- Summary dataframe Arrow serialization fix
- Edge cases (boundary values, unknown categories)
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ── Constants (mirrored from streamlit_app.py) ────────────────────────────────

FEATURE_COLUMNS = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "contract",
    "internet_service",
    "payment_method",
    "paperless_billing",
    "senior_citizen",
    "partner",
    "dependents",
    "online_security",
    "tech_support",
]

NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges"]
CATEGORICAL_FEATURES = [c for c in FEATURE_COLUMNS if c not in NUMERIC_FEATURES]


# ── Helpers (logic extracted from streamlit_app.py without Streamlit deps) ────

def build_customer_frame(
    tenure=12,
    monthly_charges=70.0,
    total_charges=840.0,
    contract="Month-to-month",
    internet_service="DSL",
    payment_method="Mailed check",
    paperless_billing="No",
    senior_citizen="No",
    partner="No",
    dependents="No",
    online_security="No",
    tech_support="No",
) -> pd.DataFrame:
    return pd.DataFrame([{
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract": contract,
        "internet_service": internet_service,
        "payment_method": payment_method,
        "paperless_billing": paperless_billing,
        "senior_citizen": senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "online_security": online_security,
        "tech_support": tech_support,
    }])


def risk_band(prob: float) -> str:
    if prob >= 0.7:
        return "High"
    if prob >= 0.4:
        return "Medium"
    return "Low"


@pytest.fixture(scope="module")
def trained_model() -> Pipeline:
    """Train the same model as in streamlit_app.py (without Streamlit cache)."""
    rng = np.random.default_rng(21)
    n = 2500

    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.56, 0.23, 0.21])
    internet = rng.choice(["DSL", "Fiber optic", "No"], size=n, p=[0.33, 0.49, 0.18])
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n, p=[0.35, 0.2, 0.23, 0.22],
    )
    paperless = rng.choice(["Yes", "No"], size=n, p=[0.65, 0.35])
    senior = rng.choice(["Yes", "No"], size=n, p=[0.16, 0.84])
    partner = rng.choice(["Yes", "No"], size=n, p=[0.54, 0.46])
    dependents = rng.choice(["Yes", "No"], size=n, p=[0.29, 0.71])

    tenure = np.select(
        [contract == "Month-to-month", contract == "One year", contract == "Two year"],
        [rng.integers(1, 28, size=n), rng.integers(8, 48, size=n), rng.integers(18, 73, size=n)],
    ).astype(int)

    base = np.select(
        [internet == "Fiber optic", internet == "DSL", internet == "No"], [82, 58, 25]
    ).astype(float)
    monthly = (
        base
        + (paperless == "Yes") * rng.uniform(2, 7, size=n)
        + (senior == "Yes") * rng.uniform(1, 8, size=n)
        + rng.normal(0, 8, size=n)
    ).clip(18, 130)

    online_sec = np.where(internet == "No", "No internet service",
                          rng.choice(["Yes", "No"], size=n, p=[0.35, 0.65]))
    tech_sup = np.where(internet == "No", "No internet service",
                        rng.choice(["Yes", "No"], size=n, p=[0.3, 0.7]))
    total = (monthly * tenure + rng.normal(0, 45, size=n)).clip(20, None)

    logit = (
        -1.25
        + 1.15 * (contract == "Month-to-month")
        + 0.75 * (internet == "Fiber optic")
        + 0.72 * (payment == "Electronic check")
        + 0.38 * (paperless == "Yes")
        + 0.35 * (senior == "Yes")
        - 0.022 * tenure
        + 0.018 * (monthly - 70)
        - 0.6 * (online_sec == "Yes")
        - 0.52 * (tech_sup == "Yes")
        - 0.3 * (partner == "Yes")
        - 0.2 * (dependents == "Yes")
    )
    churn = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    df = pd.DataFrame({
        "tenure": tenure, "monthly_charges": monthly.round(2),
        "total_charges": total.round(2), "contract": contract,
        "internet_service": internet, "payment_method": payment,
        "paperless_billing": paperless, "senior_citizen": senior,
        "partner": partner, "dependents": dependents,
        "online_security": online_sec, "tech_support": tech_sup,
        "churn": churn,
    })

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000)),
    ])
    pipeline.fit(df[FEATURE_COLUMNS], df["churn"])
    return pipeline


# ── Tests: build_customer_frame ───────────────────────────────────────────────

class TestBuildCustomerFrame:

    def test_returns_dataframe(self):
        df = build_customer_frame()
        assert isinstance(df, pd.DataFrame)

    def test_single_row(self):
        df = build_customer_frame()
        assert len(df) == 1

    def test_has_all_feature_columns(self):
        df = build_customer_frame()
        for col in FEATURE_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_numeric_types(self):
        df = build_customer_frame(tenure=24, monthly_charges=85.5, total_charges=2052.0)
        assert df["tenure"].dtype in (int, np.int64, np.int32)
        assert isinstance(df["monthly_charges"].iloc[0], float)
        assert isinstance(df["total_charges"].iloc[0], float)

    def test_values_stored_correctly(self):
        df = build_customer_frame(tenure=6, monthly_charges=55.0, contract="One year")
        assert df["tenure"].iloc[0] == 6
        assert df["monthly_charges"].iloc[0] == 55.0
        assert df["contract"].iloc[0] == "One year"

    def test_categorical_values_stored(self):
        df = build_customer_frame(
            internet_service="Fiber optic",
            payment_method="Electronic check",
            senior_citizen="Yes",
        )
        assert df["internet_service"].iloc[0] == "Fiber optic"
        assert df["payment_method"].iloc[0] == "Electronic check"
        assert df["senior_citizen"].iloc[0] == "Yes"


# ── Tests: Model structure ────────────────────────────────────────────────────

class TestModelStructure:

    def test_is_pipeline(self, trained_model):
        assert isinstance(trained_model, Pipeline)

    def test_has_preprocessor_and_classifier(self, trained_model):
        assert "preprocessor" in trained_model.named_steps
        assert "classifier" in trained_model.named_steps

    def test_classifier_is_logistic_regression(self, trained_model):
        assert isinstance(trained_model.named_steps["classifier"], LogisticRegression)

    def test_model_is_fitted(self, trained_model):
        # fitted models have classes_ attribute
        assert hasattr(trained_model.named_steps["classifier"], "classes_")

    def test_binary_classes(self, trained_model):
        classes = trained_model.named_steps["classifier"].classes_
        assert set(classes) == {0, 1}


# ── Tests: Prediction output ──────────────────────────────────────────────────

class TestPrediction:

    def test_predict_returns_0_or_1(self, trained_model):
        df = build_customer_frame()
        pred = trained_model.predict(df)[0]
        assert pred in (0, 1)

    def test_predict_proba_returns_two_columns(self, trained_model):
        df = build_customer_frame()
        proba = trained_model.predict_proba(df)
        assert proba.shape == (1, 2)

    def test_probability_between_0_and_1(self, trained_model):
        df = build_customer_frame()
        prob = trained_model.predict_proba(df)[0, 1]
        assert 0.0 <= prob <= 1.0

    def test_probabilities_sum_to_1(self, trained_model):
        df = build_customer_frame()
        proba = trained_model.predict_proba(df)[0]
        assert abs(proba.sum() - 1.0) < 1e-6

    def test_batch_prediction(self, trained_model):
        rows = [build_customer_frame() for _ in range(5)]
        batch = pd.concat(rows, ignore_index=True)
        preds = trained_model.predict(batch)
        assert len(preds) == 5
        assert all(p in (0, 1) for p in preds)


# ── Tests: High-risk vs low-risk profiles ────────────────────────────────────

class TestRiskProfiles:

    def test_high_risk_customer_has_elevated_churn_prob(self, trained_model):
        # Month-to-month + Fiber optic + Electronic check + new customer → high risk
        df = build_customer_frame(
            tenure=2,
            monthly_charges=95.0,
            total_charges=190.0,
            contract="Month-to-month",
            internet_service="Fiber optic",
            payment_method="Electronic check",
            paperless_billing="Yes",
            senior_citizen="Yes",
            online_security="No",
            tech_support="No",
        )
        prob = trained_model.predict_proba(df)[0, 1]
        assert prob > 0.5, f"Expected high-risk prob > 0.5, got {prob:.3f}"

    def test_low_risk_customer_has_low_churn_prob(self, trained_model):
        # Two-year contract + long tenure + security/support → low risk
        df = build_customer_frame(
            tenure=60,
            monthly_charges=55.0,
            total_charges=3300.0,
            contract="Two year",
            internet_service="DSL",
            payment_method="Bank transfer",
            paperless_billing="No",
            senior_citizen="No",
            partner="Yes",
            online_security="Yes",
            tech_support="Yes",
        )
        prob = trained_model.predict_proba(df)[0, 1]
        assert prob < 0.5, f"Expected low-risk prob < 0.5, got {prob:.3f}"

    def test_high_risk_prob_greater_than_low_risk(self, trained_model):
        high_risk = build_customer_frame(
            tenure=1, monthly_charges=100.0, total_charges=100.0,
            contract="Month-to-month", internet_service="Fiber optic",
            payment_method="Electronic check", senior_citizen="Yes",
        )
        low_risk = build_customer_frame(
            tenure=65, monthly_charges=40.0, total_charges=2600.0,
            contract="Two year", internet_service="No",
            payment_method="Credit card", senior_citizen="No",
            online_security="No internet service", tech_support="No internet service",
        )
        prob_high = trained_model.predict_proba(high_risk)[0, 1]
        prob_low = trained_model.predict_proba(low_risk)[0, 1]
        assert prob_high > prob_low


# ── Tests: Edge cases ─────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_minimum_tenure(self, trained_model):
        df = build_customer_frame(tenure=0, total_charges=0.0)
        prob = trained_model.predict_proba(df)[0, 1]
        assert 0.0 <= prob <= 1.0

    def test_maximum_tenure(self, trained_model):
        df = build_customer_frame(tenure=72, total_charges=9360.0)
        prob = trained_model.predict_proba(df)[0, 1]
        assert 0.0 <= prob <= 1.0

    def test_minimum_monthly_charges(self, trained_model):
        df = build_customer_frame(monthly_charges=18.0)
        prob = trained_model.predict_proba(df)[0, 1]
        assert 0.0 <= prob <= 1.0

    def test_maximum_monthly_charges(self, trained_model):
        df = build_customer_frame(monthly_charges=130.0)
        prob = trained_model.predict_proba(df)[0, 1]
        assert 0.0 <= prob <= 1.0

    def test_no_internet_service_options(self, trained_model):
        df = build_customer_frame(
            internet_service="No",
            online_security="No internet service",
            tech_support="No internet service",
        )
        prob = trained_model.predict_proba(df)[0, 1]
        assert 0.0 <= prob <= 1.0

    def test_unknown_category_handled(self, trained_model):
        # OneHotEncoder with handle_unknown='ignore' should not crash
        df = build_customer_frame(contract="Unknown contract type")
        prob = trained_model.predict_proba(df)[0, 1]
        assert 0.0 <= prob <= 1.0


# ── Tests: Risk band logic ────────────────────────────────────────────────────

class TestRiskBand:

    def test_high_band_at_0_7(self):
        assert risk_band(0.70) == "High"

    def test_high_band_above_0_7(self):
        assert risk_band(0.95) == "High"

    def test_medium_band_at_0_4(self):
        assert risk_band(0.40) == "Medium"

    def test_medium_band_between_0_4_and_0_7(self):
        assert risk_band(0.55) == "Medium"

    def test_low_band_below_0_4(self):
        assert risk_band(0.39) == "Low"

    def test_low_band_at_zero(self):
        assert risk_band(0.0) == "Low"

    def test_high_band_at_one(self):
        assert risk_band(1.0) == "High"


# ── Tests: Summary dataframe (Arrow serialization fix) ───────────────────────

class TestSummaryDataframe:

    def test_value_column_is_all_strings(self):
        df = build_customer_frame(tenure=12, monthly_charges=70.0, contract="Month-to-month")
        summary = df.T.reset_index()
        summary.columns = ["Feature", "Value"]
        summary["Value"] = summary["Value"].astype(str)
        assert summary["Value"].dtype == object
        assert all(isinstance(v, str) for v in summary["Value"])

    def test_summary_has_correct_columns(self):
        df = build_customer_frame()
        summary = df.T.reset_index()
        summary.columns = ["Feature", "Value"]
        summary["Value"] = summary["Value"].astype(str)
        assert list(summary.columns) == ["Feature", "Value"]

    def test_summary_row_count_matches_features(self):
        df = build_customer_frame()
        summary = df.T.reset_index()
        summary.columns = ["Feature", "Value"]
        assert len(summary) == len(FEATURE_COLUMNS)

    def test_no_mixed_types_after_fix(self):
        df = build_customer_frame(tenure=5, monthly_charges=65.5, contract="One year")
        summary = df.T.reset_index()
        summary.columns = ["Feature", "Value"]
        summary["Value"] = summary["Value"].astype(str)
        # All values should be strings — no int/float types left
        unique_types = {type(v) for v in summary["Value"]}
        assert unique_types == {str}
