from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_models, split_data, train_dummy, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["primary_data", "params:features", "params:split"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train", "params:features"],
                outputs="model",
                name="train_model_node",
            ),
            node(
                func=train_dummy,
                inputs=["X_train", "y_train"],
                outputs="dummy_model",
                name="train_dummy_node",
            ),
            node(
                func=evaluate_models,
                inputs=[
                    "model",
                    "dummy_model",
                    "X_test",
                    "y_test",
                    "y_train",
                    "params:features",
                    "params:split",
                    "params:data_mode",
                ],
                outputs="metrics",
                name="evaluate_models_node",
            ),
        ]
    )
