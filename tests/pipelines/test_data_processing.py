"""Testy nodów pipeline'u data_processing."""

import pandas as pd

from asi_project.pipelines.data_processing.nodes import (
    filter_training_data,
    validate_and_normalize,
)


def test_data_processing_nodes_work_on_sample(repo_root):
    sample_path = repo_root / "data" / "01_raw" / "sample.csv"
    assert sample_path.exists(), "Brak próbki sample.csv"

    df_raw = pd.read_csv(sample_path)
    df_norm = validate_and_normalize(df_raw)

    assert not df_norm.empty
    assert len(df_norm.columns) == len(df_raw.columns)

    df_filtered = filter_training_data(df_norm)
    assert not df_filtered.empty
    assert len(df_filtered) <= len(df_norm)
