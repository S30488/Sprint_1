import json
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline

from asi_project.config import get_contract, load_project_config
from asi_project.data import make_xy, split_train_test
from asi_project.metrics import build_metrics_record, compute_metrics, validate_metrics_record
from asi_project.model import build_baseline_pipeline, make_dummy
from asi_project.model_package import (
    benchmark_model_package,
    get_git_sha,
    save_model_package,
)
from asi_project.tracking import log_experiment_run


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
    X_train: pd.DataFrame | None = None,
) -> dict[str, Any]:
    config = load_project_config()
    contract = get_contract(config)
    config = replace(config, test_size=split["test_size"], random_state=split["random_state"])

    dummy_metrics = compute_metrics(y_test, dummy_model.predict(X_test))
    logreg_metrics = compute_metrics(y_test, model.predict(X_test))

    record = build_metrics_record(
        contract=contract,
        config=config,
        features=_feature_list(features),
        n_train=len(y_train),
        n_test=len(y_test),
        data_mode=data_mode,
        results={
            "dummy": dummy_metrics,
            "logistic_regression": logreg_metrics,
        },
    )
    problems = validate_metrics_record(record, contract)
    if problems:
        raise ValueError("Metryki niezgodne z kontraktem:\n" + "\n".join(problems))

    if data_mode == "full":
        git_sha = get_git_sha()
        data_fp = str(
            record.get("data_sha256")
            or record.get("fingerprint")
            or getattr(contract, "sha256", None)
            or "unknown"
        )
        allowed_feats = _feature_list(features)
        base_params = {
            "test_size": split["test_size"],
            "random_state": split["random_state"],
            "data_mode": data_mode,
            "n_train": len(y_train),
            "n_test": len(y_test),
        }

        candidates: list[tuple[str, str, Any, dict[str, Any], str]] = [
            ("baseline_dummy", "DummyClassifier", dummy_model, dummy_metrics, "baseline"),
            ("baseline_logreg", "LogisticRegression", model, logreg_metrics, "baseline"),
        ]

        if X_train is not None:
            rf_pipe = clone(model)
            rf_pipe.set_params(
                model=RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    random_state=split["random_state"],
                )
            )
            rf_pipe.fit(X_train, y_train)
            rf_metrics = compute_metrics(y_test, rf_pipe.predict(X_test))

            gb_pipe = clone(model)
            gb_pipe.set_params(
                model=GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=4,
                    random_state=split["random_state"],
                )
            )
            gb_pipe.fit(X_train, y_train)
            gb_metrics = compute_metrics(y_test, gb_pipe.predict(X_test))

            candidates.extend(
                [
                    ("random_forest", "RandomForestClassifier", rf_pipe, rf_metrics, "experiment"),
                    (
                        "gradient_boosting",
                        "GradientBoostingClassifier",
                        gb_pipe,
                        gb_metrics,
                        "experiment",
                    ),
                ]
            )

        bench_summary: dict[str, Any] = {}
        for name, m_type, estimator, m_scores, job_t in candidates:
            pkg_dir = Path("data/06_models") / name
            save_model_package(
                estimator,
                pkg_dir,
                model_name=name,
                model_type=m_type,
                allowed_features=allowed_feats,
                target_column=getattr(contract, "target", "Churn"),
                positive_class=getattr(contract, "positive_class", 1),
                metrics=m_scores,
                data_fingerprint=data_fp,
            )
            sys_stats = benchmark_model_package(pkg_dir, X_test)
            bench_summary[name] = {**m_scores, **sys_stats}

            log_experiment_run(
                run_name=f"{name}-s3",
                job_type=job_t,
                params={"model_name": name, "model_type": m_type, **base_params},
                metrics=m_scores,
                system_metrics=sys_stats,
                git_sha=git_sha,
                data_fingerprint=data_fp,
                tags=["sprint-3", job_t, name],
            )

        record["system_benchmarks"] = bench_summary

    return record


def _data_fingerprint() -> str:
    config = load_project_config()
    contract = get_contract(config)
    path = config.root / "data" / "01_raw" / contract.dataset_id / "fingerprint.json"
    if not path.exists():
        return "unknown"
    return json.loads(path.read_text(encoding="utf-8")).get("sha256", "unknown")


def train_autogluon(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    autogluon: dict[str, Any],
) -> dict[str, Any]:
    if not autogluon.get("enabled", False):
        return {"enabled": False}

    # import tylko gdy AG jest włączony (w CI nie jest)
    from autogluon.tabular import TabularPredictor

    contract = get_contract(load_project_config())
    train_df = X_train.assign(**{contract.target: y_train})
    git_sha = get_git_sha()
    data_fp = _data_fingerprint()

    results: dict[str, Any] = {}
    for name, cfg in autogluon["configs"].items():
        pkg_dir = Path(autogluon["output_dir"]) / name
        train_dir = pkg_dir.parent / f"{name}_training"
        shutil.rmtree(pkg_dir, ignore_errors=True)
        shutil.rmtree(train_dir, ignore_errors=True)

        start = time.perf_counter()
        predictor = TabularPredictor(
            label=contract.target,
            path=str(train_dir),
            eval_metric=autogluon["eval_metric"],
            problem_type="binary",
            verbosity=0,
        ).fit(
            train_df,
            presets=cfg["presets"],
            time_limit=cfg["time_limit"],
            hyperparameters=cfg.get("hyperparameters"),
        )
        fit_time_s = round(time.perf_counter() - start, 1)

        # modele do pamięci, żeby pakiet nie zależał od katalogu treningowego
        predictor.persist("all")
        scores = compute_metrics(y_test, predictor.predict(X_test))

        params = {
            "model_name": name,
            "model_type": "AutoGluonTabularPredictor",
            "presets": cfg["presets"],
            "time_limit": cfg["time_limit"],
            "eval_metric": autogluon["eval_metric"],
            "best_model": predictor.model_best,
            "n_models": len(predictor.model_names()),
            "fit_time_s": fit_time_s,
            "n_train": len(y_train),
            "n_test": len(y_test),
        }
        save_model_package(
            predictor,
            pkg_dir,
            model_name=name,
            model_type="AutoGluonTabularPredictor",
            allowed_features=list(X_train.columns),
            target_column=contract.target,
            positive_class=contract.positive_class,
            metrics=scores,
            data_fingerprint=data_fp,
            extra_metadata={"autogluon": params},
        )
        shutil.rmtree(train_dir, ignore_errors=True)
        sys_stats = benchmark_model_package(pkg_dir, X_test)

        log_experiment_run(
            run_name=f"{name}-s3",
            job_type="autogluon",
            params=params,
            metrics=scores,
            system_metrics=sys_stats,
            git_sha=git_sha,
            data_fingerprint=data_fp,
            tags=["sprint-3", "autogluon", name],
        )
        results[name] = {**params, **scores, **sys_stats}

    return results
