import pandas as pd
from typing import Any
import numpy as np
from ml_wind_turbine_anomaly_sept2026.common.mlflow_utils import load_model_by_alias


def load_champion_model(mlflow_params: dict[str, Any]) -> Any:
    """
    Load the production (champion) model from MLflow model registry.

    Wrapper that extracts config values and calls load_model_by_alias.

    Parameters
    ----------
    mlflow_params : dict[str, Any]
        MLflow config containing 'registered_model_name' and 'model_aliases'.

    Returns
    -------
    Any
        The loaded production model (MLflow pyfunc model) with scaler bundled.
    """
    registered_model_name = mlflow_params["registered_model_name"]
    production_alias = mlflow_params["model_aliases"]["production"]
    return load_model_by_alias(registered_model_name, alias=production_alias)


def predict(features_data: pd.DataFrame, champion_model: Any, predictions_column_name: list[str]) -> pd.DataFrame:
    """
    Make predictions using the champion model on new data.

    Parameters
    ----------
    features_data : pd.DataFrame
        Input features for prediction. Must have the same columns as the
        training data (excluding the target column).
    champion_model : Any
        Loaded champion model (MLflow pyfunc model) that has a `predict` method.
    predictions_column_name: str
        Names of the predictions columns
    Returns
    -------
    pd.Series
        Predicted target values for the input features.
    """
    predictions = champion_model.predict(features_data)
    return pd.DataFrame(predictions, columns=predictions_column_name)


def compute_model_errors(y_pred: pd.Series, y_true: pd.Series, anomaly_error_type: str) -> pd.DataFrame:
    """
    Compute MAPE metric from predictions and target data.

    Parameters
    ----------
    y_pred : pd.Series
        Predicted values.
    target_data : pd.Series
        Ground truth target values.

    Returns
    -------
    pd.DataFrame: DataFrame containing the error metric.
    """
    y_true = y_true.values.ravel()
    y_pred = y_pred.values.ravel()
    if anomaly_error_type == "mape":
        error = np.abs(y_true - y_pred) / (y_true + 1e-8) * 100
        error_column_name = "mape"
    return pd.DataFrame(error, columns=[error_column_name])


def compute_rolling_error(df_error: pd.DataFrame, rolling_window: int) -> pd.DataFrame:
    """
    Smooth a metric using a rolling window.

    Parameters
    ----------
    df_error : pd.DataFrame
        DataFrame containing the error metric.
    rolling_window : int
        Window size for the rolling window.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the smoothed error metric.
    """
    error_name = df_error.columns[0]
    df_error[f"rolling_{error_name}"] = df_error[error_name].rolling(window=rolling_window).median().bfill()
    return df_error


def detect_anomaly(df_error: pd.DataFrame, threshold: float, anomaly_error_type: str) -> pd.DataFrame:
    """
    Detect anomalies using a threshold.
    """
    df_anomaly = pd.DataFrame({
        "anomaly": (df_error[f"rolling_{anomaly_error_type}"] > threshold).astype(int)
    })
    return df_anomaly