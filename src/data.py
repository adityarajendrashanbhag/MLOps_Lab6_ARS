from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
import pandas as pd


TARGET_CANDIDATES = ("churn", "attrition", "exited", "target", "label")


@dataclass(frozen=True)
class DatasetBundle:
    frame: pd.DataFrame
    target_column: str | None
    source_name: str


def snake_case_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {
        column: (
            column.strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("(", "")
            .replace(")", "")
        )
        for column in df.columns
    }
    return df.rename(columns=renamed)


def _yes_no(series: pd.Series) -> pd.Series:
    mapping = {
        "yes": "Yes",
        "y": "Yes",
        "true": "Yes",
        "1": "Yes",
        "no": "No",
        "n": "No",
        "false": "No",
        "0": "No",
    }
    return (
        series.fillna("Unknown")
        .astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
        .fillna(series.fillna("Unknown").astype(str))
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _pick_target_column(df: pd.DataFrame) -> str | None:
    for candidate in TARGET_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


def _normalize_target(series: pd.Series) -> pd.Series:
    mapping = {
        "yes": 1,
        "true": 1,
        "1": 1,
        "churned": 1,
        "left": 1,
        "no": 0,
        "false": 0,
        "0": 0,
        "stayed": 0,
    }
    normalized = (
        series.astype(str).str.strip().str.lower().map(mapping).where(series.notna(), np.nan)
    )
    if normalized.notna().any():
        return normalized.astype("Int64")
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.round().clip(0, 1).astype("Int64")
    return series


def standardize_churn_frame(df: pd.DataFrame) -> DatasetBundle:
    frame = snake_case_columns(df.copy())

    rename_map = {
        "customerid": "customer_id",
        "monthlycharges": "monthly_charges",
        "totalcharges": "total_charges",
        "internetservice": "internet_service",
        "paymentmethod": "payment_method",
        "paperlessbilling": "paperless_billing",
        "seniorcitizen": "senior_citizen",
        "onlinesecurity": "online_security",
        "onlinebackup": "online_backup",
        "deviceprotection": "device_protection",
        "techsupport": "tech_support",
        "streamingtv": "streaming_tv",
        "streamingmovies": "streaming_movies",
        "multiplelines": "multiple_lines",
    }
    frame = frame.rename(columns=rename_map)

    for column in ("tenure", "monthly_charges", "total_charges"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for column in (
        "partner",
        "dependents",
        "phone_service",
        "multiple_lines",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
        "paperless_billing",
    ):
        if column in frame.columns:
            frame[column] = _yes_no(frame[column])

    if "senior_citizen" in frame.columns:
        frame["senior_citizen"] = _yes_no(frame["senior_citizen"])

    target_column = _pick_target_column(frame)
    if target_column is not None:
        frame[target_column] = _normalize_target(frame[target_column])

    return DatasetBundle(frame=frame, target_column=target_column, source_name="uploaded csv")


def generate_demo_churn_data(n_rows: int = 1200, random_state: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    gender = rng.choice(["Female", "Male"], size=n_rows)
    senior = rng.choice(["No", "Yes"], size=n_rows, p=[0.84, 0.16])
    partner = rng.choice(["No", "Yes"], size=n_rows, p=[0.46, 0.54])
    dependents = rng.choice(["No", "Yes"], size=n_rows, p=[0.7, 0.3])
    phone_service = rng.choice(["No", "Yes"], size=n_rows, p=[0.1, 0.9])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        rng.choice(["No", "Yes"], size=n_rows, p=[0.5, 0.5]),
    )
    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"],
        size=n_rows,
        p=[0.34, 0.5, 0.16],
    )
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_rows,
        p=[0.56, 0.23, 0.21],
    )
    paperless = rng.choice(["No", "Yes"], size=n_rows, p=[0.36, 0.64])
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n_rows,
        p=[0.34, 0.19, 0.24, 0.23],
    )

    tenure = np.select(
        [contract == "Month-to-month", contract == "One year", contract == "Two year"],
        [
            rng.integers(1, 28, size=n_rows),
            rng.integers(10, 50, size=n_rows),
            rng.integers(20, 73, size=n_rows),
        ],
    ).astype(int)

    base_charge = np.select(
        [internet_service == "Fiber optic", internet_service == "DSL", internet_service == "No"],
        [78, 58, 24],
    ).astype(float)
    monthly_charges = (
        base_charge
        + (paperless == "Yes") * rng.uniform(2, 8, size=n_rows)
        + (multiple_lines == "Yes") * rng.uniform(5, 18, size=n_rows)
        + (senior == "Yes") * rng.uniform(1, 10, size=n_rows)
        + rng.normal(0, 8, size=n_rows)
    ).clip(18, 130)

    def service_flag(base_prob: float, no_internet_value: str = "No internet service") -> np.ndarray:
        raw = rng.choice(["No", "Yes"], size=n_rows, p=[1 - base_prob, base_prob])
        return np.where(internet_service == "No", no_internet_value, raw)

    online_security = service_flag(0.34)
    online_backup = service_flag(0.42)
    device_protection = service_flag(0.41)
    tech_support = service_flag(0.31)
    streaming_tv = service_flag(0.49)
    streaming_movies = service_flag(0.48)

    total_charges = (monthly_charges * tenure + rng.normal(0, 45, size=n_rows)).clip(20, None)

    risk_score = (
        -1.15
        + 1.1 * (contract == "Month-to-month")
        + 0.72 * (payment_method == "Electronic check")
        + 0.62 * (internet_service == "Fiber optic")
        + 0.45 * (paperless == "Yes")
        + 0.35 * (senior == "Yes")
        - 0.02 * tenure
        + 0.018 * (monthly_charges - 70)
        - 0.65 * np.isin(online_security, ["Yes"]).astype(float)
        - 0.55 * np.isin(tech_support, ["Yes"]).astype(float)
        - 0.8 * (contract == "Two year")
        - 0.35 * (partner == "Yes")
    )
    churn = rng.binomial(1, _sigmoid(risk_score))

    return pd.DataFrame(
        {
            "customer_id": [f"C{idx:05d}" for idx in range(1, n_rows + 1)],
            "gender": gender,
            "senior_citizen": senior,
            "partner": partner,
            "dependents": dependents,
            "tenure": tenure,
            "phone_service": phone_service,
            "multiple_lines": multiple_lines,
            "internet_service": internet_service,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "contract": contract,
            "paperless_billing": paperless,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges.round(2),
            "total_charges": total_charges.round(2),
            "churn": churn,
        }
    )


def get_demo_bundle(n_rows: int = 1200, random_state: int = 7) -> DatasetBundle:
    frame = generate_demo_churn_data(n_rows=n_rows, random_state=random_state)
    return DatasetBundle(frame=frame, target_column="churn", source_name="bundled demo dataset")


def load_uploaded_bundle(raw_bytes: bytes) -> DatasetBundle:
    return standardize_churn_frame(pd.read_csv(BytesIO(raw_bytes)))
