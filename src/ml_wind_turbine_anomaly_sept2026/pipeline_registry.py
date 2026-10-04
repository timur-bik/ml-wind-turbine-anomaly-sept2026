from kedro.pipeline import Pipeline
from ml_wind_turbine_anomaly_sept2026.pipelines.feature_eng.pipeline import create_pipeline as create_feature_eng_pipeline  
from ml_wind_turbine_anomaly_sept2026.pipelines.training.pipeline import create_pipeline as create_training_pipeline
from ml_wind_turbine_anomaly_sept2026.pipelines.inference.pipeline import create_pipeline as create_inference_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    feature_eng_pipeline = create_feature_eng_pipeline()
    training_pipeline = create_training_pipeline()
    inference_pipeline = create_inference_pipeline()

    return {
        "__default__": feature_eng_pipeline + training_pipeline,
        "training": feature_eng_pipeline + training_pipeline,
        "inference": feature_eng_pipeline + inference_pipeline,
    }
