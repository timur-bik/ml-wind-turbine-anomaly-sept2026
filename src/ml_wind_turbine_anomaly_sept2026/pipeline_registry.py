from kedro.pipeline import Pipeline
from ml_wind_turbine_anomaly_sept2026.pipelines.feature_eng.pipeline import create_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    feature_eng_pipeline = create_pipeline()

    return {
        "__default__": feature_eng_pipeline,
        "feature_eng": feature_eng_pipeline
    }
