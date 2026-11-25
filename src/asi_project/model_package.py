from __future__ import annotations

import json
import platform
import subprocess
import time
import tracemalloc
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def get_git_sha() -> str:
    """Zwraca skrócony lub pełny SHA bieżącego commita Git."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def get_library_versions(packages: list[str] | None = None) -> dict[str, str]:
    """Zbiera wersje kluczowych bibliotek środowiska."""
    target_packages = packages or [
        "scikit-learn",
        "pandas",
        "numpy",
        "joblib",
        "kedro",
        "wandb",
        "autogluon.tabular",
    ]
    versions: dict[str, str] = {"python": platform.python_version()}
    for pkg in target_packages:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "not-installed"
    return versions


def get_dir_size_bytes(path: str | Path) -> int:
    """Oblicza łączny rozmiar pliku lub katalogu w bajtach."""
    target = Path(path)
    if target.is_file():
        return target.stat().st_size
    return sum(f.stat().st_size for f in target.rglob("*") if f.is_file())


def save_model_package(
    model: Any,
    package_dir: str | Path,
    *,
    model_name: str,
    model_type: str,
    allowed_features: list[str],
    target_column: str,
    positive_class: Any,
    metrics: dict[str, Any],
    data_fingerprint: str,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Zapisuje pakiet modelu: plik modelu oraz metadata.json."""
    out_dir = Path(package_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.joblib"
    joblib.dump(model, model_path)

    metadata: dict[str, Any] = {
        "model_name": model_name,
        "model_type": model_type,
        "allowed_features": list(allowed_features),
        "target_column": target_column,
        "positive_class": positive_class,
        "metrics": metrics,
        "git_sha": get_git_sha(),
        "data_fingerprint": data_fingerprint,
        "library_versions": get_library_versions(),
        "created_at": datetime.now(UTC).isoformat(),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return out_dir


def load_model_package(package_dir: str | Path) -> tuple[Any, dict[str, Any]]:
    """Wczytuje model oraz metadata.json z katalogu pakietu."""
    pkg_dir = Path(package_dir)
    metadata_path = pkg_dir / "metadata.json"
    model_path = pkg_dir / "model.joblib"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Brak pliku metadata.json w {pkg_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Brak pliku model.joblib w {pkg_dir}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model = joblib.load(model_path)
    return model, metadata


def benchmark_model_package(
    package_dir: str | Path,
    X_sample: pd.DataFrame,
    n_repeats: int = 10,
) -> dict[str, float]:
    """Mierzy rozmiar artefaktu, pamięć RAM przy ładowaniu oraz czas predykcji."""
    size_bytes = get_dir_size_bytes(package_dir)

    tracemalloc.start()
    t_load_start = time.perf_counter()
    model, metadata = load_model_package(package_dir)
    load_time_ms = (time.perf_counter() - t_load_start) * 1000.0
    _, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    features = metadata.get("allowed_features", list(X_sample.columns))
    X_eval = X_sample[features]

    # Rozgrzewka
    _ = model.predict(X_eval)

    t_pred_start = time.perf_counter()
    for _ in range(n_repeats):
        _ = model.predict(X_eval)
    total_pred_ms = (time.perf_counter() - t_pred_start) * 1000.0

    avg_batch_ms = total_pred_ms / max(n_repeats, 1)
    per_row_ms = avg_batch_ms / max(len(X_eval), 1)

    return {
        "artifact_size_kb": round(size_bytes / 1024.0, 3),
        "load_peak_ram_kb": round(peak_mem_bytes / 1024.0, 3),
        "load_time_ms": round(load_time_ms, 3),
        "predict_batch_ms": round(avg_batch_ms, 3),
        "predict_per_row_ms": round(per_row_ms, 4),
    }
