from kedro.pipeline import Pipeline, node, pipeline

from .nodes import filter_training_data, validate_and_normalize


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=validate_and_normalize,
                inputs="raw_data",
                outputs="normalized_data",
                name="validate_and_normalize_node",
            ),
            node(
                func=filter_training_data,
                inputs="normalized_data",
                outputs="primary_data",
                name="filter_training_data_node",
            ),
        ]
    )
