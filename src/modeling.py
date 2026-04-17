from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CLASSIFIERS: dict[str, object] = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}

_DROP = {"customer_id"}


@dataclass
class ModelResult:
    model_name: str
    metrics: pd.DataFrame
    feature_importance: pd.DataFrame
    confusion: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray
    auc: float
    pipeline: Pipeline
    train_rows: int
    test_rows: int


def _preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
                num_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                cat_cols,
            ),
        ]
    )


def train_model(
    df: pd.DataFrame,
    target: str,
    model_name: str = "Logistic Regression",
    test_size: float = 0.25,
) -> ModelResult:
    X = df.drop(columns=[c for c in (_DROP | {target}) if c in df.columns])
    y = pd.to_numeric(df[target], errors="coerce").dropna().astype(int)
    X = X.loc[y.index]

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    pipeline = Pipeline([
        ("prep", _preprocessor(num_cols, cat_cols)),
        ("clf", CLASSIFIERS[model_name]),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = pd.DataFrame([
        {"Metric": "Accuracy",  "Value": round(accuracy_score(y_test, y_pred), 4)},
        {"Metric": "Precision", "Value": round(precision_score(y_test, y_pred, zero_division=0), 4)},
        {"Metric": "Recall",    "Value": round(recall_score(y_test, y_pred, zero_division=0), 4)},
        {"Metric": "F1 Score",  "Value": round(f1_score(y_test, y_pred, zero_division=0), 4)},
        {"Metric": "ROC AUC",   "Value": round(roc_auc_score(y_test, y_prob), 4)},
    ])

    prep = pipeline.named_steps["prep"]
    cat_feature_names = prep.named_transformers_["cat"].named_steps["enc"].get_feature_names_out(cat_cols).tolist()
    all_features = num_cols + cat_feature_names
    clf = pipeline.named_steps["clf"]

    if hasattr(clf, "coef_"):
        importance_vals = np.abs(clf.coef_[0])
    elif hasattr(clf, "feature_importances_"):
        importance_vals = clf.feature_importances_
    else:
        importance_vals = np.zeros(len(all_features))

    fi = (
        pd.DataFrame({"feature": all_features, "importance": importance_vals})
        .nlargest(15, "importance")
        .reset_index(drop=True)
    )

    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    return ModelResult(
        model_name=model_name,
        metrics=metrics,
        feature_importance=fi,
        confusion=cm,
        fpr=fpr,
        tpr=tpr,
        auc=auc,
        pipeline=pipeline,
        train_rows=len(X_train),
        test_rows=len(X_test),
    )


def predict_customer(pipeline: Pipeline, customer: dict) -> tuple[int, float]:
    prob = float(pipeline.predict_proba(pd.DataFrame([customer]))[0, 1])
    return (1 if prob >= 0.5 else 0), prob
