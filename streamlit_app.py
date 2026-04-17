from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data import DatasetBundle, get_demo_bundle, load_uploaded_bundle
from src.modeling import ModelArtifacts, train_churn_model
from src.weak_supervision import add_weak_labels, apply_weak_supervision


st.set_page_config(
    page_title="Churn Weak Supervision Lab",
    page_icon="CL",
    layout="wide",
)


CUSTOM_CSS = """
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(255, 230, 204, 0.75), transparent 30%),
            radial-gradient(circle at top right, rgba(191, 223, 255, 0.7), transparent 28%),
            linear-gradient(180deg, #fff9f2 0%, #f7fbff 100%);
    }
    .hero {
        padding: 1.25rem 1.5rem;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(246,250,255,0.96));
        border: 1px solid rgba(33, 57, 88, 0.08);
        box-shadow: 0 20px 45px rgba(52, 88, 130, 0.10);
        margin-bottom: 1.25rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.4rem;
        color: #17324d;
        letter-spacing: -0.03em;
    }
    .hero p {
        margin: 0.4rem 0 0 0;
        color: #41566e;
        font-size: 1rem;
    }
    .pill {
        display: inline-block;
        padding: 0.32rem 0.7rem;
        border-radius: 999px;
        background: #eaf4ff;
        color: #1e4f7b;
        font-size: 0.86rem;
        margin-right: 0.45rem;
    }
</style>
"""


@st.cache_data(show_spinner=False)
def get_demo_data(n_rows: int, seed: int) -> DatasetBundle:
    return get_demo_bundle(n_rows=n_rows, random_state=seed)


@st.cache_data(show_spinner=False)
def load_uploaded_data(raw_bytes: bytes) -> DatasetBundle:
    return load_uploaded_bundle(raw_bytes)


@st.cache_data(show_spinner=False)
def run_pipeline(
    frame: pd.DataFrame, target_column: str | None, epochs: int
) -> tuple[pd.DataFrame, object, ModelArtifacts | None]:
    weak_artifacts = apply_weak_supervision(frame, label_model_epochs=epochs)
    enriched = add_weak_labels(frame, weak_artifacts)
    model_artifacts = train_churn_model(enriched, target_column=target_column)
    return enriched, weak_artifacts, model_artifacts


def render_header(
    bundle: DatasetBundle,
    enriched: pd.DataFrame,
    weak_artifacts: object,
    model_artifacts: ModelArtifacts | None,
) -> None:
    target_text = bundle.target_column or "not detected"
    st.markdown(
        f"""
        <div class="hero">
            <span class="pill">Source: {bundle.source_name}</span>
            <span class="pill">Rows: {len(bundle.frame):,}</span>
            <span class="pill">Target: {target_text}</span>
            <h1>Customer Churn Weak Supervision Lab</h1>
            <p>Programmatic labeling, probabilistic churn signals, and a lightweight classifier wrapped in one Streamlit app.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Weak coverage", f"{weak_artifacts.coverage:.1%}")
    metric_columns[1].metric("Conflict rate", f"{weak_artifacts.conflict_rate:.1%}")
    metric_columns[2].metric("Overlap rate", f"{weak_artifacts.overlap_rate:.1%}")
    metric_columns[3].metric("Avg churn score", f"{enriched['churn_probability'].mean():.2f}")

    if model_artifacts is not None and not model_artifacts.metrics.empty:
        metrics_lookup = dict(zip(model_artifacts.metrics["metric"], model_artifacts.metrics["value"]))
        model_cols = st.columns(3)
        model_cols[0].metric("Holdout F1", f"{metrics_lookup.get('f1', 0):.3f}")
        model_cols[1].metric("Holdout ROC AUC", f"{metrics_lookup.get('roc_auc', 0):.3f}")
        model_cols[2].metric("Weak training rows", f"{model_artifacts.weak_training_rows:,}")


def overview_tab(enriched: pd.DataFrame, target_column: str | None) -> None:
    left, right = st.columns([1.1, 0.9])
    with left:
        st.subheader("Dataset preview")
        st.dataframe(enriched.head(25), use_container_width=True)

    with right:
        st.subheader("Weak label distribution")
        weak_counts = (
            enriched["weak_label_name"].value_counts(dropna=False).rename_axis("label").reset_index(name="rows")
        )
        fig = px.bar(
            weak_counts,
            x="label",
            y="rows",
            color="label",
            color_discrete_sequence=["#5b8def", "#ff7f50", "#92c353"],
        )
        fig.update_layout(height=360, showlegend=False, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

        if target_column is not None:
            st.subheader("Observed churn")
            churn_rate = enriched[target_column].dropna().mean()
            st.metric("Gold churn rate", f"{churn_rate:.1%}")


def rules_tab(weak_artifacts: object) -> None:
    st.subheader("Rulebook")
    rules = pd.DataFrame(
        [
            ("lf_month_to_month_fiber_risk", "Flags costly month-to-month fiber plans as churn-prone."),
            ("lf_new_customer_echeck_risk", "Flags new electronic-check customers using paperless billing."),
            ("lf_senior_high_bill_risk", "Flags senior customers with very high monthly charges."),
            ("lf_long_term_secure_customer", "Marks long-tenure customers with longer contracts and tech support as safe."),
            ("lf_protection_bundle_safe", "Marks customers with security and device protection as more stable."),
            ("lf_low_charge_no_internet_safe", "Marks low-charge customers without internet service as retain."),
        ],
        columns=["labeling_function", "idea"],
    )
    st.dataframe(rules, use_container_width=True, hide_index=True)

    st.subheader("LF analysis")
    st.dataframe(weak_artifacts.lf_summary, use_container_width=True, hide_index=True)


def weak_labels_tab(enriched: pd.DataFrame, weak_artifacts: object) -> None:
    st.subheader("Probabilistic labels")
    preview_cols = [
        column
        for column in [
            "customer_id",
            "contract",
            "internet_service",
            "tenure",
            "monthly_charges",
            "payment_method",
            "weak_label_name",
            "churn_probability",
        ]
        if column in enriched.columns
    ]
    st.dataframe(
        enriched.sort_values("churn_probability", ascending=False)[preview_cols].head(30),
        use_container_width=True,
        hide_index=True,
    )

    chart_col, matrix_col = st.columns([1.0, 1.2])
    with chart_col:
        fig = px.histogram(
            enriched,
            x="churn_probability",
            nbins=24,
            color="weak_label_name",
            color_discrete_map={"Churn": "#ff7f50", "Retain": "#5b8def", "Abstain": "#92c353"},
        )
        fig.update_layout(height=360, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with matrix_col:
        label_matrix = pd.DataFrame(
            weak_artifacts.label_matrix[:20],
            columns=weak_artifacts.lf_summary["lf_name"].tolist(),
        )
        st.caption("First 20 rows of the labeling matrix")
        st.dataframe(label_matrix, use_container_width=True)


def model_tab(model_artifacts: ModelArtifacts | None) -> None:
    st.subheader("Classifier")
    if model_artifacts is None:
        st.info(
            "Model evaluation is available when the dataset includes a usable churn target column with enough labeled rows."
        )
        return

    left, right = st.columns([0.85, 1.15])
    with left:
        st.dataframe(model_artifacts.metrics, use_container_width=True, hide_index=True)

    with right:
        fig = px.bar(
            model_artifacts.feature_importance.iloc[::-1],
            x="weight",
            y="feature",
            orientation="h",
            color="weight",
            color_continuous_scale=["#5b8def", "#ffb36b", "#ff7f50"],
        )
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=20, b=0), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)


def slices_tab(model_artifacts: ModelArtifacts | None, enriched: pd.DataFrame) -> None:
    st.subheader("Slice analysis")
    if model_artifacts is None or model_artifacts.slice_metrics.empty:
        st.info("Slice metrics appear here when the app can train and score a holdout model against a gold churn target.")
    else:
        st.dataframe(model_artifacts.slice_metrics, use_container_width=True, hide_index=True)

    if "contract" in enriched.columns and "churn_probability" in enriched.columns:
        fig = px.box(
            enriched,
            x="contract",
            y="churn_probability",
            color="contract",
            color_discrete_sequence=["#5b8def", "#92c353", "#ff7f50"],
        )
        fig.update_layout(height=360, showlegend=False, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)


def sidebar_controls() -> tuple[str, int, int, int, bytes | None]:
    st.sidebar.header("Lab Controls")
    data_source = st.sidebar.radio("Dataset source", ["Demo dataset", "Upload CSV"], index=0)
    sample_size = st.sidebar.slider("Demo rows", min_value=400, max_value=3000, value=1200, step=100)
    seed = st.sidebar.slider("Random seed", min_value=1, max_value=99, value=7)
    epochs = st.sidebar.slider("LabelModel epochs", min_value=100, max_value=600, value=300, step=50)
    uploaded_file = st.sidebar.file_uploader("Upload churn CSV", type=["csv"])
    return data_source, sample_size, seed, epochs, uploaded_file.getvalue() if uploaded_file else None


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    data_source, sample_size, seed, epochs, uploaded_bytes = sidebar_controls()

    if data_source == "Upload CSV" and uploaded_bytes is not None:
        bundle = load_uploaded_data(uploaded_bytes)
    else:
        bundle = get_demo_data(sample_size, seed)

    enriched, weak_artifacts, model_artifacts = run_pipeline(bundle.frame, bundle.target_column, epochs)
    render_header(bundle, enriched, weak_artifacts, model_artifacts)

    tabs = st.tabs(["Overview", "Rules", "Weak Labels", "Model", "Slices"])
    with tabs[0]:
        overview_tab(enriched, bundle.target_column)
    with tabs[1]:
        rules_tab(weak_artifacts)
    with tabs[2]:
        weak_labels_tab(enriched, weak_artifacts)
    with tabs[3]:
        model_tab(model_artifacts)
    with tabs[4]:
        slices_tab(model_artifacts, enriched)


if __name__ == "__main__":
    main()
