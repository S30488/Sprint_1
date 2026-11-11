from dataclasses import replace
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from asi_project.config import get_contract, load_project_config
from asi_project.data import make_xy, split_train_test
from asi_project.metrics import build_metrics_record, compute_metrics, validate_metrics_record
from asi_project.model import build_baseline_pipeline, make_dummy


def _feature_list(features: dict[str, list[str]]) -> list[str]:
    return [*features["numeric"], *features["categorical"]]


def split_data(
    primary_data: pd.DataFrame,
    features: dict[str, list[str]],
    split: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    contract = get_contract(load_project_config())
    X, y = make_xy(primary_data, contract, _feature_list(features))
    return split_train_test(X, y, contract, split["test_size"], split["random_state"])


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    features: dict[str, list[str]],
) -> Pipeline:
    model = build_baseline_pipeline(features["numeric"], features["categorical"])
    return model.fit(X_train, y_train)


def train_dummy(X_train: pd.DataFrame, y_train: pd.Series) -> BaseEstimator:
    return make_dummy().fit(X_train, y_train)


def evaluate_models(
    model: Pipeline,
    dummy_model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_train: pd.Series,
    features: dict[str, list[str]],
    split: dict[str, Any],
    data_mode: str,
) -> dict[str, Any]:
    config = load_project_config()
    contract = get_contract(config)
    # metrics.json ma opisywać split faktycznie użyty w pipeline (z parameters.yml)
    config = replace(config, test_size=split["test_size"], random_state=split["random_state"])

    record = build_metrics_record(
        contract=contract,
        config=config,
        features=_feature_list(features),
        n_train=len(y_train),
        n_test=len(y_test),
        data_mode=data_mode,
        results={
            "dummy": compute_metrics(y_test, dummy_model.predict(X_test)),
            "logistic_regression": compute_metrics(y_test, model.predict(X_test)),
        },
    )
    problems = validate_metrics_record(record, contract)
    if problems:
        raise ValueError("Metryki niezgodne z kontraktem:\n" + "\n".join(problems))
    return record
