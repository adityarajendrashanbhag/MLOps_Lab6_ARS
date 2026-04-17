from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_EXCLUDE = {"customer_id", "weak_label", "weak_label_name", "churn_probability"}


@dataclass
class ModelArtifacts:
    metrics: pd.DataFrame
    feature_importance: pd.DataFrame
    score_frame: pd.DataFrame
    slice_metrics: pd.DataFrame
    weak_training_rows: int


def _build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [column for column in df.columns if column not in numeric_cols]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )


def _compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    return pd.DataFrame(
        {"metric": list(metrics.keys()), "value": [round(value, 4) for value in metrics.values()]}
    )


def _slice_metrics(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    slice_definitions = {
        "month_to_month": df.get("contract", pd.Series(index=df.index, dtype=object)).eq("Month-to-month"),
        "fiber_optic": df.get("internet_service", pd.Series(index=df.index, dtype=object)).eq("Fiber optic"),
        "senior_citizen": df.get("senior_citizen", pd.Series(index=df.index, dtype=object)).eq("Yes"),
        "high_monthly_charges": df.get("monthly_charges", pd.Series(index=df.index, dtype=float)).fillna(0) >= 90,
    }

    rows = []
    for slice_name, mask in slice_definitions.items():
        subset = df.loc[mask]
        if subset.empty or subset[target_column].nunique(dropna=True) < 2:
            continue
        rows.append(
            {
                "slice": slice_name,
                "rows": len(subset),
                "churn_rate": round(float(subset[target_column].mean()), 4),
                "avg_model_score": round(float(subset["model_score"].mean()), 4),
            }
        )
    return pd.DataFrame(rows)


def train_churn_model(
    df: pd.DataFrame,
    target_column: str | None,
) -> ModelArtifacts | None:
    if target_column is None or target_column not in df.columns:
        return None

    labeled = df.loc[df["weak_label"].isin([0, 1])].copy()
    labeled = labeled.dropna(subset=[target_column])
    if len(labeled) < 80 or labeled["weak_label"].nunique() < 2 or labeled[target_column].nunique() < 2:
        return None

    feature_columns = [column for column in labeled.columns if column not in DEFAULT_EXCLUDE | {target_column}]
    X = labeled[feature_columns]
    y_weak = labeled["weak_label"].astype(int)
    y_gold = labeled[target_column].astype(int)

    X_train, X_test, y_train, _, _, y_gold_test = train_test_split(
        X,
        y_weak,
        y_gold,
        test_size=0.25,
        random_state=13,
        stratify=y_gold,
    )

    pipeline = Pipeline(
        [
            ("preprocessor", _build_preprocessor(X_train)),
            ("model", LogisticRegression(max_iter=1500, class_weight="balanced")),
        ]
    )
    pipeline.fit(X_train, y_train)

    scores = pipeline.predict_proba(X_test)[:, 1]
    predictions = (scores >= 0.5).astype(int)
    metrics = _compute_metrics(y_gold_test, predictions, scores)

    encoded_features = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefficients = pipeline.named_steps["model"].coef_[0]
    feature_importance = (
        pd.DataFrame({"feature": encoded_features, "weight": coefficients})
        .assign(abs_weight=lambda frame: frame["weight"].abs())
        .sort_values("abs_weight", ascending=False)
        .drop(columns="abs_weight")
        .head(15)
        .reset_index(drop=True)
    )

    score_frame = df.copy()
    score_frame["model_score"] = pipeline.predict_proba(df[feature_columns])[:, 1]
    score_frame["predicted_churn"] = (score_frame["model_score"] >= 0.5).astype(int)
    slice_metrics = _slice_metrics(score_frame.dropna(subset=[target_column]), target_column)

    return ModelArtifacts(
        metrics=metrics,
        feature_importance=feature_importance,
        score_frame=score_frame,
        slice_metrics=slice_metrics,
        weak_training_rows=len(labeled),
    )
