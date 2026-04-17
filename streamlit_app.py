from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(page_title="Customer Churn Predictor", page_icon="CP", layout="centered")

st.markdown(
    """
    <style>
    /* ── Page background ───────────────────────────────────────────────── */
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(255, 229, 204, 0.85), transparent 28%),
            radial-gradient(circle at top right, rgba(208, 231, 255, 0.75), transparent 26%),
            linear-gradient(180deg, #fffaf4 0%, #f8fbff 100%);
    }

    /* ── Hero card ─────────────────────────────────────────────────────── */
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 22px;
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(20, 49, 82, 0.08);
        box-shadow: 0 18px 40px rgba(44, 86, 129, 0.10);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        color: #16324a;
        font-size: 2.2rem;
        letter-spacing: -0.03em;
    }
    .hero p {
        margin: 0.45rem 0 0 0;
        color: #4a6076;
        font-size: 1rem;
    }

    /* ── Info card ─────────────────────────────────────────────────────── */
    .info-card {
        padding: 0.9rem 1rem;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid rgba(20, 49, 82, 0.08);
        margin-bottom: 1rem;
        color: #29445b;
    }

    /* ── Form labels ───────────────────────────────────────────────────── */
    label, .stSlider label, .stNumberInput label, .stSelectbox label {
        color: #17324a !important;
        font-weight: 600;
    }

    /* ── Selectbox — dark background, light text ───────────────────────── */
    div[data-baseweb="select"] > div {
        background-color: #1e2d3d !important;
        color: #f0f4f8 !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
    }
    div[data-baseweb="select"] span {
        color: #f0f4f8 !important;
    }
    div[data-baseweb="select"] input {
        color: #f0f4f8 !important;
        -webkit-text-fill-color: #f0f4f8 !important;
    }
    div[data-baseweb="select"] svg {
        fill: #f0f4f8 !important;
    }

    /* ── Dropdown open list ─────────────────────────────────────────────── */
    ul[data-baseweb="menu"] {
        background-color: #1e2d3d !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
    }
    ul[data-baseweb="menu"] li {
        background-color: #1e2d3d !important;
        color: #f0f4f8 !important;
    }
    ul[data-baseweb="menu"] li:hover,
    ul[data-baseweb="menu"] li[aria-selected="true"] {
        background-color: #2d4a6b !important;
        color: #ffffff !important;
    }

    /* ── Number input — dark background, light text ─────────────────────── */
    div[data-testid="stNumberInput"] input {
        background-color: #1e2d3d !important;
        color: #f0f4f8 !important;
        -webkit-text-fill-color: #f0f4f8 !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
    }
    div[data-testid="stNumberInput"] button {
        background-color: #2d4a6b !important;
        color: #f0f4f8 !important;
    }

    /* ── Metrics — light background, dark text ────────────────────────── */
    [data-testid="stMetricLabel"] { color: #17324a !important; }
    [data-testid="stMetricValue"] { color: #17324a !important; }
    [data-testid="stMetricDelta"] { color: #17324a !important; }

    /* ── Result cards (churn / stay) ───────────────────────────────────── */
    .result-churn {
        padding: 0.85rem 1rem;
        border-radius: 10px;
        background: #fde8e8;
        border-left: 4px solid #c0392b;
        color: #1a1a1a !important;
        font-weight: 600;
        font-size: 0.97rem;
        margin-bottom: 0.6rem;
    }
    .result-stay {
        padding: 0.85rem 1rem;
        border-radius: 10px;
        background: #e8f5e9;
        border-left: 4px solid #2e7d32;
        color: #1a1a1a !important;
        font-weight: 600;
        font-size: 0.97rem;
        margin-bottom: 0.6rem;
    }

    /* ── st.write / st.markdown text ───────────────────────────────────── */
    .stMarkdown p  { color: #17324a !important; }
    .stMarkdown h3 { color: #17324a !important; }

    /* ── st.table ──────────────────────────────────────────────────────── */
    table          { color: #17324a !important; }
    table th       { color: #17324a !important; background-color: #ddeeff !important; }
    table td       { color: #17324a !important; background-color: #f7fbff !important; }

    /* ── st.dataframe ──────────────────────────────────────────────────── */
    [data-testid="stDataFrame"] { color: #17324a; }
    </style>
    """,
    unsafe_allow_html=True,
)


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


@st.cache_resource(show_spinner=False)
def train_demo_model() -> Pipeline:
    rng = np.random.default_rng(21)
    n_rows = 2500

    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n_rows, p=[0.56, 0.23, 0.21])
    internet_service = rng.choice(["DSL", "Fiber optic", "No"], size=n_rows, p=[0.33, 0.49, 0.18])
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n_rows,
        p=[0.35, 0.2, 0.23, 0.22],
    )
    paperless_billing = rng.choice(["Yes", "No"], size=n_rows, p=[0.65, 0.35])
    senior_citizen = rng.choice(["Yes", "No"], size=n_rows, p=[0.16, 0.84])
    partner = rng.choice(["Yes", "No"], size=n_rows, p=[0.54, 0.46])
    dependents = rng.choice(["Yes", "No"], size=n_rows, p=[0.29, 0.71])

    tenure = np.select(
        [contract == "Month-to-month", contract == "One year", contract == "Two year"],
        [
            rng.integers(1, 28, size=n_rows),
            rng.integers(8, 48, size=n_rows),
            rng.integers(18, 73, size=n_rows),
        ],
    ).astype(int)

    base_charge = np.select(
        [internet_service == "Fiber optic", internet_service == "DSL", internet_service == "No"],
        [82, 58, 25],
    ).astype(float)
    monthly_charges = (
        base_charge
        + (paperless_billing == "Yes") * rng.uniform(2, 7, size=n_rows)
        + (senior_citizen == "Yes") * rng.uniform(1, 8, size=n_rows)
        + rng.normal(0, 8, size=n_rows)
    ).clip(18, 130)

    online_security = np.where(
        internet_service == "No",
        "No internet service",
        rng.choice(["Yes", "No"], size=n_rows, p=[0.35, 0.65]),
    )
    tech_support = np.where(
        internet_service == "No",
        "No internet service",
        rng.choice(["Yes", "No"], size=n_rows, p=[0.3, 0.7]),
    )

    total_charges = (monthly_charges * tenure + rng.normal(0, 45, size=n_rows)).clip(20, None)

    churn_logit = (
        -1.25
        + 1.15 * (contract == "Month-to-month")
        + 0.75 * (internet_service == "Fiber optic")
        + 0.72 * (payment_method == "Electronic check")
        + 0.38 * (paperless_billing == "Yes")
        + 0.35 * (senior_citizen == "Yes")
        - 0.022 * tenure
        + 0.018 * (monthly_charges - 70)
        - 0.6 * (online_security == "Yes")
        - 0.52 * (tech_support == "Yes")
        - 0.3 * (partner == "Yes")
        - 0.2 * (dependents == "Yes")
    )
    churn_probability = 1 / (1 + np.exp(-churn_logit))
    churn = rng.binomial(1, churn_probability)

    train_df = pd.DataFrame(
        {
            "tenure": tenure,
            "monthly_charges": monthly_charges.round(2),
            "total_charges": total_charges.round(2),
            "contract": contract,
            "internet_service": internet_service,
            "payment_method": payment_method,
            "paperless_billing": paperless_billing,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "online_security": online_security,
            "tech_support": tech_support,
            "churn": churn,
        }
    )

    numeric_features = ["tenure", "monthly_charges", "total_charges"]
    categorical_features = [column for column in FEATURE_COLUMNS if column not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000)),
        ]
    )
    model.fit(train_df[FEATURE_COLUMNS], train_df["churn"])
    return model


def build_customer_frame(
    tenure: int,
    monthly_charges: float,
    total_charges: float,
    contract: str,
    internet_service: str,
    payment_method: str,
    paperless_billing: str,
    senior_citizen: str,
    partner: str,
    dependents: str,
    online_security: str,
    tech_support: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
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
            }
        ]
    )


model = train_demo_model()

st.markdown(
    """
    <div class="hero">
        <h1>Customer Churn Predictor</h1>
        <p>Enter telecom customer details and predict churn with one click.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-card">
        This is a simple Streamlit lab app. It uses a lightweight logistic regression model trained on synthetic
        telecom churn data so you can focus on the user interface and prediction workflow.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
        monthly_charges = st.slider("Monthly Charges ($)", min_value=18.0, max_value=130.0, value=70.0, step=0.5)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        )
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

    with col2:
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            value=float(max(tenure, 1) * monthly_charges),
            step=10.0,
        )

    predict = st.form_submit_button("Predict Churn", type="primary", use_container_width=True)

if predict:
    customer_df = build_customer_frame(
        tenure=tenure,
        monthly_charges=monthly_charges,
        total_charges=total_charges,
        contract=contract,
        internet_service=internet_service,
        payment_method=payment_method,
        paperless_billing=paperless_billing,
        senior_citizen=senior_citizen,
        partner=partner,
        dependents=dependents,
        online_security=online_security,
        tech_support=tech_support,
    )

    churn_probability = float(model.predict_proba(customer_df)[0, 1])
    prediction = int(churn_probability >= 0.5)

    left, right = st.columns([1, 1])
    with left:
        if prediction == 1:
            st.markdown(
                '<div class="result-churn">⚠️ Prediction: Customer is likely to churn</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-stay">✅ Prediction: Customer is likely to stay</div>',
                unsafe_allow_html=True,
            )
        st.metric("Churn Probability", f"{churn_probability:.1%}")

    with right:
        risk_band = "High" if churn_probability >= 0.7 else "Medium" if churn_probability >= 0.4 else "Low"
        st.metric("Risk Band", risk_band)
        st.markdown("**Submitted customer details**")
        summary_df = customer_df.T.reset_index()
        summary_df.columns = ["Feature", "Value"]
        summary_df["Value"] = summary_df["Value"].astype(str)
        st.table(summary_df)
