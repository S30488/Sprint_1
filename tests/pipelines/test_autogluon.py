"""Testy nodu AutoGluon (bez pełnego treningu i bez W&B)."""

import pytest
import yaml

from asi_project.data import filter_training_rows, normalize_input
from asi_project.model_package import load_model_package
from asi_project.pipelines.data_science.nodes import split_data, train_autogluon


@pytest.fixture
def params(repo_root):
    path = repo_root / "conf" / "base" / "parameters.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture
def split_parts(project, params):
    primary = filter_training_rows(normalize_input(project.sample), project.contract)
    return split_data(primary, params["features"], params["split"])


def test_autogluon_disabled_does_not_train(split_parts):
    X_train, X_test, y_train, y_test = split_parts
    assert train_autogluon(X_train, y_train, X_test, y_test, {"enabled": False}) == {
        "enabled": False
    }


def test_ci_parameters_disable_autogluon(repo_root, params):
    ci = yaml.safe_load((repo_root / "conf" / "ci" / "parameters.yml").read_text(encoding="utf-8"))
    assert ci["autogluon"]["enabled"] is False
    for cfg in params["autogluon"]["configs"].values():
        assert cfg["time_limit"] > 0


def test_autogluon_small_model_package_roundtrip(split_parts, tmp_path, monkeypatch):
    pytest.importorskip("autogluon.tabular")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    X_train, X_test, y_train, y_test = split_parts

    # mały model testowy: tylko LightGBM, kilka sekund
    ag_params = {
        "enabled": True,
        "eval_metric": "f1",
        "output_dir": str(tmp_path),
        "configs": {
            "ag_test": {
                "presets": "medium_quality",
                "time_limit": 30,
                "hyperparameters": {"GBM": {"num_boost_round": 20}},
            }
        },
    }
    results = train_autogluon(X_train, y_train, X_test, y_test, ag_params)

    assert 0.0 <= results["ag_test"]["f1"] <= 1.0
    assert results["ag_test"]["artifact_size_kb"] > 0
    # katalog treningowy usunięty, pakiet musi działać sam
    assert not (tmp_path / "ag_test_training").exists()

    model, metadata = load_model_package(tmp_path / "ag_test")
    assert metadata["allowed_features"] == list(X_train.columns)
    proba = model.predict_proba(X_test.head(5))
    assert proba.shape == (5, 2)
