from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from snorkel.labeling import LFAnalysis, PandasLFApplier, labeling_function
from snorkel.labeling.model import LabelModel, MajorityLabelVoter


ABSTAIN = -1
RETAIN = 0
CHURN = 1


def _text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _num(value: object) -> float:
    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


@labeling_function()
def lf_month_to_month_fiber_risk(row: pd.Series) -> int:
    if (
        _text(row.get("contract")) == "month-to-month"
        and _text(row.get("internet_service")) == "fiber optic"
        and _num(row.get("monthly_charges")) >= 80
    ):
        return CHURN
    return ABSTAIN


@labeling_function()
def lf_new_customer_echeck_risk(row: pd.Series) -> int:
    if (
        _num(row.get("tenure")) <= 12
        and _text(row.get("payment_method")) == "electronic check"
        and _text(row.get("paperless_billing")) == "yes"
    ):
        return CHURN
    return ABSTAIN


@labeling_function()
def lf_senior_high_bill_risk(row: pd.Series) -> int:
    if _text(row.get("senior_citizen")) == "yes" and _num(row.get("monthly_charges")) >= 95:
        return CHURN
    return ABSTAIN


@labeling_function()
def lf_long_term_secure_customer(row: pd.Series) -> int:
    if (
        _num(row.get("tenure")) >= 48
        and _text(row.get("contract")) in {"one year", "two year"}
        and _text(row.get("tech_support")) == "yes"
    ):
        return RETAIN
    return ABSTAIN


@labeling_function()
def lf_protection_bundle_safe(row: pd.Series) -> int:
    if _text(row.get("online_security")) == "yes" and _text(row.get("device_protection")) == "yes":
        return RETAIN
    return ABSTAIN


@labeling_function()
def lf_low_charge_no_internet_safe(row: pd.Series) -> int:
    if _text(row.get("internet_service")) == "no" and _num(row.get("monthly_charges")) <= 45:
        return RETAIN
    return ABSTAIN


def build_lfs() -> list:
    return [
        lf_month_to_month_fiber_risk,
        lf_new_customer_echeck_risk,
        lf_senior_high_bill_risk,
        lf_long_term_secure_customer,
        lf_protection_bundle_safe,
        lf_low_charge_no_internet_safe,
    ]


@dataclass
class WeakSupervisionArtifacts:
    lf_summary: pd.DataFrame
    label_matrix: np.ndarray
    majority_labels: np.ndarray
    weak_labels: np.ndarray
    weak_probabilities: np.ndarray
    coverage: float
    conflict_rate: float
    overlap_rate: float
    model_type: str
    labeled_mask: np.ndarray


def apply_weak_supervision(df: pd.DataFrame, label_model_epochs: int = 300) -> WeakSupervisionArtifacts:
    lfs = build_lfs()
    applier = PandasLFApplier(lfs=lfs)
    label_matrix = applier.apply(df=df)
    summary = LFAnalysis(L=label_matrix, lfs=lfs).lf_summary()

    majority_model = MajorityLabelVoter(cardinality=2)
    majority_labels = majority_model.predict(L=label_matrix, tie_break_policy="abstain")

    coverage_mask = label_matrix != ABSTAIN
    row_has_label = coverage_mask.any(axis=1)
    row_overlap = coverage_mask.sum(axis=1) > 1
    conflict_mask = np.array(
        [len({vote for vote in row if vote != ABSTAIN}) > 1 for row in label_matrix]
    )

    observed_votes = label_matrix[row_has_label]
    observed_votes = observed_votes[observed_votes != ABSTAIN]
    use_label_model = row_has_label.sum() >= 50 and np.unique(observed_votes).size > 1

    if use_label_model:
        label_model = LabelModel(cardinality=2, verbose=False)
        label_model.fit(L_train=label_matrix, n_epochs=label_model_epochs, log_freq=100, seed=13)
        weak_labels = label_model.predict(L=label_matrix, tie_break_policy="abstain")
        weak_probabilities = label_model.predict_proba(L=label_matrix)
        model_type = "Snorkel LabelModel"
    else:
        weak_labels = majority_labels
        positive_prob = np.where(majority_labels == CHURN, 0.75, 0.25)
        positive_prob = np.where(majority_labels == ABSTAIN, 0.5, positive_prob)
        weak_probabilities = np.column_stack([1 - positive_prob, positive_prob])
        model_type = "MajorityLabelVoter fallback"

    labeled_mask = weak_labels != ABSTAIN

    return WeakSupervisionArtifacts(
        lf_summary=summary.reset_index(names="lf_name"),
        label_matrix=label_matrix,
        majority_labels=majority_labels,
        weak_labels=weak_labels,
        weak_probabilities=weak_probabilities,
        coverage=float(row_has_label.mean()),
        conflict_rate=float(conflict_mask.mean()),
        overlap_rate=float(row_overlap.mean()),
        model_type=model_type,
        labeled_mask=labeled_mask,
    )


def add_weak_labels(df: pd.DataFrame, artifacts: WeakSupervisionArtifacts) -> pd.DataFrame:
    enriched = df.copy()
    enriched["weak_label"] = artifacts.weak_labels
    enriched["weak_label_name"] = enriched["weak_label"].map({0: "Retain", 1: "Churn", -1: "Abstain"})
    enriched["churn_probability"] = artifacts.weak_probabilities[:, 1]
    return enriched
