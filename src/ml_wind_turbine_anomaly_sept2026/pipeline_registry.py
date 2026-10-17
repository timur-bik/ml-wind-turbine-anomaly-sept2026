from kedro.pipeline import Pipeline

from ml_wind_turbine_anomaly_sept2026.pipelines.feature_eng.pipeline import (
    feat_eng_pipeline_inference,
    feat_eng_pipeline_training,
)
from ml_wind_turbine_anomaly_sept2026.pipelines.inference.pipeline import (
    create_pipeline as create_inference_pipeline,
)
from ml_wind_turbine_anomaly_sept2026.pipelines.monitoring.pipeline import (
    create_monitoring_pipeline,
)
from ml_wind_turbine_anomaly_sept2026.pipelines.training.pipeline import (
    create_pipeline as create_training_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    feature_eng_pipeline_training = feat_eng_pipeline_training()
    feature_eng_pipeline_inference = feat_eng_pipeline_inference()
    training_pipeline = create_training_pipeline()
    inference_pipeline = create_inference_pipeline()
    monitoring_pipeline = create_monitoring_pipeline()
    return {
        "__default__": feature_eng_pipeline_training + training_pipeline,
        "training": feature_eng_pipeline_training + training_pipeline,
        "inference": feature_eng_pipeline_inference + inference_pipeline,
        "monitoring": monitoring_pipeline,
    }
