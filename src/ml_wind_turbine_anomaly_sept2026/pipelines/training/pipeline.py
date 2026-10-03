from kedro.pipeline import Pipeline, node
from .nodes import train_test_split, tune_hyperparameters, fit_best_model

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=train_test_split,
            inputs=["x_data", "y_data", "params:training_pipeline"],
            outputs=["x_train", "y_train", "x_test", "y_test"],
        ),
        node(
            func=tune_hyperparameters,
            inputs=["x_train", "y_train", "params:training_pipeline"],
            outputs="tuning_results",
        ),
        node(
            func=fit_best_model,
            inputs=["x_train", "y_train", "x_test", "y_test", "params:training_pipeline", "tuning_results"],
            outputs="training_results",
        ),
    ])