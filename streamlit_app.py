from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data import DatasetBundle, get_demo_bundle, load_uploaded_bundle
from src.modeling import CLASSIFIERS, ModelResult, predict_customer, train_model


st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #f0f4f8 0%, #e8f0f7 100%); }
    .hero {
        background: linear-gradient(135deg, #1a237e 0%, #1565c0 60%, #0288d1 100%);
        color: white;
        padding: 1.75rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(26,35,126,0.18);
    }
    .hero h1 { margin: 0; font-size: 2rem; letter-spacing: -0.02em; }
    .hero p  { margin: 0.4rem 0 0; opacity: 0.88; font-size: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── cached helpers ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _demo(n: int, seed: int) -> DatasetBundle:
    return get_demo_bundle(n_rows=n, random_state=seed)


@st.cache_data(show_spinner=False)
def _upload(raw: bytes) -> DatasetBundle:
    return load_uploaded_bundle(raw)


@st.cache_data(show_spinner=False)
def _train(_df: pd.DataFrame, target: str, model_name: str, test_size: float) -> ModelResult:
    return train_model(_df, target, model_name, test_size)


# ── sidebar ───────────────────────────────────────────────────────────────────

def sidebar() -> tuple:
    st.sidebar.title("⚙️ Controls")
    source = st.sidebar.radio("Data source", ["Demo dataset", "Upload CSV"])
    n_rows, seed, uploaded_bytes = 1000, 42, None

    if source == "Demo dataset":
        n_rows = st.sidebar.slider("Sample size", 300, 3000, 1000, 100)
        seed   = st.sidebar.slider("Random seed", 1, 99, 42)
    else:
        f = st.sidebar.file_uploader("Upload churn CSV", type=["csv"])
        if f:
            uploaded_bytes = f.getvalue()

    st.sidebar.divider()
    model_name = st.sidebar.selectbox("Classifier", list(CLASSIFIERS))
    test_pct   = st.sidebar.slider("Test split %", 10, 40, 25)

    return source, n_rows, seed, uploaded_bytes, model_name, test_pct / 100


# ── Tab 1 — Data Explorer ─────────────────────────────────────────────────────

def tab_explore(df: pd.DataFrame, target: str) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total customers", f"{len(df):,}")
    c2.metric("Features", df.shape[1] - 2)
    c3.metric("Churn rate", f"{df[target].mean():.1%}" if target in df.columns else "—")
    c4.metric("Missing values", int(df.isnull().sum().sum()))

    st.subheader("Dataset preview")
    st.dataframe(df.head(25), use_container_width=True)

    st.divider()
    l, r = st.columns(2)

    with l:
        st.subheader("Churn rate by Contract")
        if "contract" in df.columns and target in df.columns:
            g = df.groupby("contract")[target].mean().reset_index(name="churn_rate")
            fig = px.bar(
                g, x="contract", y="churn_rate", color="contract",
                text=g["churn_rate"].map("{:.1%}".format),
                color_discrete_sequence=["#1565c0", "#ff5722", "#43a047"],
            )
            fig.update_layout(showlegend=False, height=300, yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

    with r:
        st.subheader("Churn rate by Internet Service")
        if "internet_service" in df.columns and target in df.columns:
            g = df.groupby("internet_service")[target].mean().reset_index(name="churn_rate")
            fig = px.bar(
                g, x="internet_service", y="churn_rate", color="internet_service",
                text=g["churn_rate"].map("{:.1%}".format),
                color_discrete_sequence=["#7b1fa2", "#ff9800", "#00acc1"],
            )
            fig.update_layout(showlegend=False, height=300, yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

    l2, r2 = st.columns(2)

    with l2:
        st.subheader("Tenure distribution")
        if "tenure" in df.columns:
            fig = px.histogram(
                df, x="tenure",
                color=target if target in df.columns else None,
                color_discrete_map={1: "#ef5350", 0: "#42a5f5"},
                nbins=36, barmode="overlay", opacity=0.75,
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    with r2:
        st.subheader("Monthly charges distribution")
        if "monthly_charges" in df.columns:
            fig = px.histogram(
                df, x="monthly_charges",
                color=target if target in df.columns else None,
                color_discrete_map={1: "#ef5350", 0: "#42a5f5"},
                nbins=36, barmode="overlay", opacity=0.75,
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)


# ── Tab 2 — Model Performance ─────────────────────────────────────────────────

def tab_model(result: ModelResult) -> None:
    st.subheader(f"Classifier: {result.model_name}")
    st.caption(f"Trained on {result.train_rows:,} rows · evaluated on {result.test_rows:,} rows")

    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown("**Performance metrics**")
        st.dataframe(result.metrics, use_container_width=True, hide_index=True)

    with r1c2:
        st.markdown("**Confusion matrix**")
        fig = px.imshow(
            result.confusion, text_auto=True, color_continuous_scale="Blues",
            x=["Predicted: Retain", "Predicted: Churn"],
            y=["Actual: Retain",    "Actual: Churn"],
        )
        fig.update_layout(height=280, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown("**ROC Curve**")
        fig = go.Figure([
            go.Scatter(
                x=result.fpr, y=result.tpr, mode="lines",
                name=f"AUC = {result.auc:.3f}",
                line=dict(color="#1565c0", width=2.5),
            ),
            go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                name="Random baseline",
                line=dict(color="gray", dash="dash"),
            ),
        ])
        fig.update_layout(
            height=320,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with r2c2:
        st.markdown("**Top 15 Feature Importance**")
        fi = result.feature_importance.iloc[::-1]
        fig = px.bar(
            fi, x="importance", y="feature", orientation="h",
            color="importance", color_continuous_scale="Blues",
        )
        fig.update_layout(
            height=320, margin=dict(l=0, r=0, t=20, b=0),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Tab 3 — Predict Customer ──────────────────────────────────────────────────

def tab_predict(result: ModelResult) -> None:
    st.subheader("Predict churn for a single customer")

    c1, c2, c3 = st.columns(3)

    with c1:
        tenure          = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 130.0, 65.0, 0.5)
        total_charges   = st.number_input(
            "Total Charges ($)", value=float(monthly_charges * max(tenure, 1)), min_value=0.0
        )

    with c2:
        contract         = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        payment_method   = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        )

    with c3:
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner        = st.selectbox("Partner",        ["No", "Yes"])
        dependents     = st.selectbox("Dependents",     ["No", "Yes"])

    with st.expander("Additional Services"):
        s1, s2, s3 = st.columns(3)
        phone_service     = s1.selectbox("Phone Service",     ["Yes", "No"])
        online_security   = s2.selectbox("Online Security",   ["No", "Yes", "No internet service"])
        tech_support      = s3.selectbox("Tech Support",      ["No", "Yes", "No internet service"])
        online_backup     = s1.selectbox("Online Backup",     ["No", "Yes", "No internet service"])
        device_protection = s2.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        streaming_tv      = s3.selectbox("Streaming TV",      ["No", "Yes", "No internet service"])
        streaming_movies  = s1.selectbox("Streaming Movies",  ["No", "Yes", "No internet service"])
        paperless_billing = s2.selectbox("Paperless Billing", ["Yes", "No"])
        gender            = s3.selectbox("Gender",            ["Male", "Female"])

    if st.button("Predict Churn", type="primary"):
        customer = dict(
            gender=gender, senior_citizen=senior_citizen, partner=partner,
            dependents=dependents, tenure=tenure, phone_service=phone_service,
            internet_service=internet_service, online_security=online_security,
            online_backup=online_backup, device_protection=device_protection,
            tech_support=tech_support, streaming_tv=streaming_tv,
            streaming_movies=streaming_movies, contract=contract,
            paperless_billing=paperless_billing, payment_method=payment_method,
            monthly_charges=monthly_charges, total_charges=total_charges,
        )
        label, prob = predict_customer(result.pipeline, customer)

        res_l, res_r = st.columns([1, 2])
        with res_l:
            if label == 1:
                st.error("⚠️ High churn risk")
            else:
                st.success("✅ Likely to stay")
            st.metric("Churn probability", f"{prob:.1%}")

        with res_r:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                number={"suffix": "%"},
                title={"text": "Churn Risk Score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#ef5350" if label == 1 else "#43a047"},
                    "steps": [
                        {"range": [0,  33], "color": "#e8f5e9"},
                        {"range": [33, 66], "color": "#fff9c4"},
                        {"range": [66, 100], "color": "#ffebee"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.75, "value": 50,
                    },
                },
            ))
            fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)


# ── Tab 4 — Insights ──────────────────────────────────────────────────────────

def tab_insights(df: pd.DataFrame, target: str) -> None:
    if target not in df.columns:
        st.info("Insights require a churn target column in your dataset.")
        return

    st.subheader("Churn Rate by Segment")

    segments = {
        "contract":        "Contract Type",
        "internet_service": "Internet Service",
        "payment_method":  "Payment Method",
        "senior_citizen":  "Senior Citizen",
    }

    cols = st.columns(2)
    for idx, (col_name, label) in enumerate(segments.items()):
        if col_name not in df.columns:
            continue
        g = df.groupby(col_name)[target].mean().reset_index(name="churn_rate")
        fig = px.bar(
            g, x=col_name, y="churn_rate", color="churn_rate",
            color_continuous_scale=["#43a047", "#ff9800", "#ef5350"],
            text=g["churn_rate"].map("{:.1%}".format),
            title=label,
        )
        fig.update_layout(
            height=270, showlegend=False, coloraxis_showscale=False,
            yaxis_tickformat=".0%", margin=dict(l=0, r=0, t=40, b=0),
        )
        with cols[idx % 2]:
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Key Risk Factors")

    risks = []
    if "contract" in df.columns:
        v = df[df["contract"] == "Month-to-month"][target].mean()
        risks.append(f"Month-to-month contracts: **{v:.1%}** churn rate")
    if "internet_service" in df.columns:
        v = df[df["internet_service"] == "Fiber optic"][target].mean()
        risks.append(f"Fiber optic customers: **{v:.1%}** churn rate")
    if "payment_method" in df.columns:
        v = df[df["payment_method"].str.contains("Electronic", na=False)][target].mean()
        risks.append(f"Electronic check users: **{v:.1%}** churn rate")
    if "senior_citizen" in df.columns:
        mask = df["senior_citizen"].isin(["Yes", 1, "1"])
        if mask.any():
            v = df.loc[mask, target].mean()
            risks.append(f"Senior citizens: **{v:.1%}** churn rate")
    if "tenure" in df.columns:
        v = df[df["tenure"] <= 12][target].mean()
        risks.append(f"First-year customers (tenure ≤ 12 months): **{v:.1%}** churn rate")

    for r in risks:
        st.markdown(f"- {r}")

    st.divider()
    st.subheader("Monthly Charges vs Tenure")
    if "tenure" in df.columns and "monthly_charges" in df.columns:
        sample = df.sample(min(len(df), 600), random_state=42)
        fig = px.scatter(
            sample, x="tenure", y="monthly_charges",
            color=target if target in df.columns else None,
            color_discrete_map={1: "#ef5350", 0: "#42a5f5"},
            labels={"tenure": "Tenure (months)", "monthly_charges": "Monthly Charges ($)"},
            opacity=0.6,
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>📡 Telecom Customer Churn Predictor</h1>
            <p>Explore data · train a classifier · predict which customers are at risk of leaving</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    source, n_rows, seed, uploaded_bytes, model_name, test_size = sidebar()

    if source == "Upload CSV" and uploaded_bytes:
        bundle = _upload(uploaded_bytes)
    else:
        bundle = _demo(n_rows, seed)

    df     = bundle.frame
    target = bundle.target_column or "churn"

    with st.spinner("Training model…"):
        result = _train(df, target, model_name, test_size)

    tabs = st.tabs(["🔍 Data Explorer", "🤖 Model Performance", "🎯 Predict Customer", "💡 Insights"])
    with tabs[0]:
        tab_explore(df, target)
    with tabs[1]:
        tab_model(result)
    with tabs[2]:
        tab_predict(result)
    with tabs[3]:
        tab_insights(df, target)


if __name__ == "__main__":
    main()
