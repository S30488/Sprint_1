"""Testy nodów pipeline'u data_science."""

import pytest
import yaml
from sklearn.pipeline import Pipeline

from asi_project.data import filter_training_rows, normalize_input
from asi_project.metrics import validate_metrics_record
from asi_project.pipelines.data_science.nodes import (
    evaluate_models,
    split_data,
    train_dummy,
    train_model,
)
from asi_project.pipelines.data_science.pipeline import create_pipeline


@pytest.fixture
def params(repo_root):
    path = repo_root / "conf" / "base" / "parameters.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture
def primary_data(project):
    return filter_training_rows(normalize_input(project.sample), project.contract)


@pytest.fixture
def split_parts(primary_data, params):
    return split_data(primary_data, params["features"], params["split"])


def test_parameters_match_project_yaml(params, project_config):
    assert params["split"]["test_size"] == project_config.test_size
    assert params["split"]["random_state"] == project_config.random_state
    assert params["features"]["numeric"] == list(project_config.numeric_features)
    assert params["features"]["categorical"] == list(project_config.categorical_features)


def test_split_data_structure(split_parts, primary_data, params):
    X_train, X_test, y_train, y_test = split_parts
    features = params["features"]["numeric"] + params["features"]["categorical"]

    assert list(X_train.columns) == features
    assert len(X_train) + len(X_test) == len(primary_data)
    assert len(X_train) == len(y_train) and len(X_test) == len(y_test)
    assert set(y_train.unique()) == {0, 1} and set(y_test.unique()) == {0, 1}
    assert X_train.index.intersection(X_test.index).empty


def test_split_data_is_reproducible(primary_data, params):
    first = split_data(primary_data, params["features"], params["split"])
    second = split_data(primary_data, params["features"], params["split"])
    assert first[1].index.equals(second[1].index)


def test_train_model_returns_fitted_pipeline(split_parts, params):
    X_train, X_test, y_train, _ = split_parts
    model = train_model(X_train, y_train, params["features"])

    assert isinstance(model, Pipeline)
    proba = model.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)


def test_evaluate_models_returns_valid_metrics(split_parts, params, project):
    X_train, X_test, y_train, y_test = split_parts
    model = train_model(X_train, y_train, params["features"])
    dummy = train_dummy(X_train, y_train)

    record = evaluate_models(
        model, dummy, X_test, y_test, y_train, params["features"], params["split"], "sample"
    )

    assert validate_metrics_record(record, project.contract) == []
    assert record["data_mode"] == "sample"
    assert record["split"]["n_test"] == len(y_test)
    assert set(record["models"]) == {"dummy", "logistic_regression"}


def test_pipeline_inputs_and_outputs():
    ds_pipeline = create_pipeline()
    assert ds_pipeline.inputs() == {
        "primary_data",
        "params:features",
        "params:split",
        "params:data_mode",
    }
    assert {"model", "metrics"} <= ds_pipeline.all_outputs()
