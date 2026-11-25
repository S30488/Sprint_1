from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import wandb


def _load_env_file(env_path: str | Path = ".env") -> None:
    """Wczytuje zmienne z pliku .env do os.environ (bez zewnętrznych bibliotek)."""
    path = Path(env_path)
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def log_experiment_run(
    *,
    run_name: str,
    job_type: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    system_metrics: dict[str, float] | None = None,
    git_sha: str,
    data_fingerprint: str,
    tags: list[str] | None = None,
    package_dir: str | Path | None = None,
    artifact_name: str | None = None,
    artifact_aliases: list[str] | None = None,
) -> str | None:
    """Inicjuje run W&B, loguje parametry, metryki i opcjonalnie artefakt modelu."""
    _load_env_file()

    project = os.getenv("WANDB_PROJECT", "asi-churn-s3")
    entity = os.getenv("WANDB_ENTITY") or None
    mode = os.getenv("WANDB_MODE", "online")

    config_payload: dict[str, Any] = {
        **params,
        "git_sha": git_sha,
        "data_fingerprint": data_fingerprint,
    }

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        job_type=job_type,
        config=config_payload,
        tags=tags or [],
        mode=mode,
        reinit="finish_previous",
    )

    if run is None:
        return None

    try:
        flat_metrics: dict[str, Any] = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                flat_metrics[k] = v
        if system_metrics:
            flat_metrics.update(system_metrics)

        wandb.log(flat_metrics)
        run.summary.update(flat_metrics)
        run.summary["git_sha"] = git_sha
        run.summary["data_fingerprint"] = data_fingerprint

        if package_dir is not None and artifact_name is not None:
            artifact = wandb.Artifact(
                name=artifact_name,
                type="model",
                metadata={
                    "git_sha": git_sha,
                    "data_fingerprint": data_fingerprint,
                    **flat_metrics,
                },
            )
            artifact.add_dir(str(package_dir))
            run.log_artifact(artifact, aliases=artifact_aliases or [])

        return run.id
    finally:
        run.finish()
