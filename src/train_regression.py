# src/train_regression.py

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from .utils import (
    configure_logging,
    get_models_path,
    get_raw_data_path,
    get_reports_path,
)

logger = logging.getLogger(__name__)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))



def load_data(filename: str = "train.csv") -> pd.DataFrame:
    """Leser Kaggle House Prices train.csv fra data/raw/."""
    path = get_raw_data_path() / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Fant ikke {path}. Last ned fra Kaggle og legg i data/raw/"
        )
    df = pd.read_csv(path)
    logger.info("Data shape: %s", df.shape)
    return df


def split_features_target(df: pd.DataFrame):
    """Skiller ut X og y. Target er SalePrice."""
    if "SalePrice" not in df.columns:
        raise ValueError("Kolonnen 'SalePrice' finnes ikke i datasettet.")
    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice"])
    return X, y


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list, list]:
    """Bygger ColumnTransformer for numeriske og kategoriske features."""
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


def get_models():
    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=400, random_state=42, n_jobs=-1
        ),
    }
    if HAS_XGB:
        models["xgboost"] = XGBRegressor(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            tree_method="auto",
        )
    return models


def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def save_metrics(results: list[dict]):
    reports_dir = get_reports_path()
    out = reports_dir / "metrics.txt"
    lines = []
    for r in results:
        lines.append(f"Model: {r['model']}")
        lines.append(f"RMSE: {r['rmse']:.2f}")
        lines.append(f"MAE : {r['mae']:.2f}")
        lines.append(f"R^2 : {r['r2']:.4f}")
        lines.append("-" * 40)
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Lagret metrikk-rapport til %s", out)


def save_best_model(pipeline: Pipeline, name: str):
    models_dir = get_models_path()
    path = models_dir / f"best_model_{name}.joblib"
    joblib.dump(pipeline, path)
    logger.info("Lagret beste modell til %s", path)


def main():
    configure_logging()
    logger.info("Starter trening for House Price Prediction...")

    # 1) Load
    df = load_data()

    # 2) Split
    X, y = split_features_target(df)

    # 3) Preprocessor
    preprocessor, num_cols, cat_cols = build_preprocessor(X)

    # 4) Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info("Train: %s, Test: %s", X_train.shape, X_test.shape)

    # 5) Models
    models = get_models()

    # 6) Train/eval
    all_results = []
    best_rmse = np.inf
    best_name = None
    best_pipe = None

    for name, model in models.items():
        logger.info("Trener modell: %s", name)
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        res = evaluate(y_test, y_pred)
        res["model"] = name
        all_results.append(res)

        logger.info("Resultater %s | RMSE=%.2f MAE=%.2f R2=%.4f",
                    name, res["rmse"], res["mae"], res["r2"])

        if res["rmse"] < best_rmse:
            best_rmse = res["rmse"]
            best_name = name
            best_pipe = pipe

    # 7) Save
    save_metrics(all_results)
    if best_pipe is not None:
        save_best_model(best_pipe, best_name)
        logger.info("Beste modell: %s (RMSE=%.2f)", best_name, best_rmse)

    logger.info("Ferdig!")


if __name__ == "__main__":
    main()
