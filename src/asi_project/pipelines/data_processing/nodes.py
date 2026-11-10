import pandas as pd

from asi_project.config import get_contract, load_project_config
from asi_project.data import (
    assert_valid,
    filter_training_rows,
    normalize_column_names,
    normalize_input,
)


def validate_and_normalize(df_raw: pd.DataFrame) -> pd.DataFrame:
    config = load_project_config()
    contract = get_contract(config)

    df = normalize_column_names(df_raw)
    df = normalize_input(df)
    assert_valid(df, contract, config)
    return df


def filter_training_data(df: pd.DataFrame) -> pd.DataFrame:
    config = load_project_config()
    contract = get_contract(config)

    return filter_training_rows(df, contract)
