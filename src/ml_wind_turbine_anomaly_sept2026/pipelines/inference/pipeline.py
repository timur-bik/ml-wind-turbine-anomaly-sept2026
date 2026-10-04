from kedro.pipeline import Pipeline, node
from .nodes import load_champion_model, predict, compute_model_errors, compute_rolling_error, detect_anomaly


def create_pipeline(**kwargs) -> Pipeline:
    """Create the inference pipeline."""
    return Pipeline([
        node(
            func=load_champion_model,
            inputs=["params:mlflow"],
            outputs="champion_model",
        ),
        node(
            func=predict,
            inputs=["x_data", "champion_model", "params:inference_pipeline.predictions_column_name"],
            outputs="predictions",
        ),
        node(
            func=compute_model_errors,
            inputs=["predictions", "y_data", "params:inference_pipeline.anomaly_error_type"],
            outputs="model_errors",
        ),
        node(
            func=compute_rolling_error,
            inputs=["model_errors", "params:inference_pipeline.rolling_window"],
            outputs="rolling_errors",
        ),
        node(
            func=detect_anomaly,
            inputs=[
                "rolling_errors", 
                "params:inference_pipeline.anomaly_threshold",
                "params:inference_pipeline.anomaly_error_type"
            ],
            outputs="anomalies",
        ),
    ])