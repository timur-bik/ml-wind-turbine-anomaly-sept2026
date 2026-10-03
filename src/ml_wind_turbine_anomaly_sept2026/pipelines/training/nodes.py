import pandas as pd
from typing import Any
import optuna
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor as RF
from ml_wind_turbine_anomaly_sept2026.common.metrics import compute_metrics
from ml_wind_turbine_anomaly_sept2026.pipelines.training.utils import eval_model, objective


def train_test_split(
    features: pd.DataFrame,
    target: pd.Series,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Explicit temporal train / val / test split using index boundaries.
    """
    test_idx = int(features.shape[0] * (1 - params["test_fraction"]))
    x_train, x_test = features.iloc[:test_idx].copy(), features.iloc[test_idx:].copy()
    y_train, y_test = target.iloc[:test_idx].copy(), target.iloc[test_idx:].copy()

    return x_train, y_train, x_test, y_test



def tune_hyperparameters(
    x_train: pd.DataFrame, y_train: pd.Series, params: dict[str, Any]
) -> dict[str, Any]:
    """
    Train a model by optimizing hyperparameters using Optuna.
    Parameters
    ----------
    x_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target values.
    params : Dict[str, Any]
        Configuration dictionary containing:
        - 'optuna_search': dict with keys:
            - 'n_trials': int, number of optimization trials to run
            - 'model': str, model name (e.g., 'CatBoost', 'RF')
            - 'model_params': dict, model-specific parameter search spaces
        - 'n_folds': int, number of CV folds (default: 3)

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'best_params': Dict[str, Any], best hyperparameters found during optimization
        - 'cv_results': Dict[str, Any], cross-validation results from best model:
            - 'cv_mae': float, average MAE over CV folds
            - 'cv_rmse': float, average RMSE over CV folds
            - 'cv_mape': float, average MAPE over CV folds
    """
    seed = 42
    # 1) Create the study and optimize CV MAPE via objective()
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(
        lambda trial: objective(trial, x_train, y_train, params),
        n_trials=params["optuna_search"]["n_trials"],
    )

    # 2) Get best parameters
    best_params = study.best_params

    # 3) Re-evaluate best model to get full CV results
    model_name = params["optuna_search"]["model"]

    cv_results = eval_model(
        x_train=x_train,
        y_train=y_train,
        n_splits=params["n_folds"],
        model_name=model_name,
        model_params=best_params,
        seed=seed,
    )

    return {
        "best_params": best_params,
        "cv_metrics": {
            "cv_mae": cv_results["cv_mae"],
            "cv_rmse": cv_results["cv_rmse"],
            "cv_mape": cv_results["cv_mape"],
        },
    }


def fit_best_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict[str, Any],
    tuning_results: dict[str, Any],
) -> dict[str, Any]:
    """
    Trains a model using optimized hyperparameters found during hyperparameter tuning.

    Parameters
    ----------
    x_train : pd.DataFrame
        Training features. Should contain all feature columns used during
        hyperparameter optimization.
    y_train : pd.Series
        Training target values. Must have the same length as x_train.
    params : Dict[str, Any]
        Configuration dictionary containing:
        - 'optuna_search': dict with keys:
            - 'model': str, model name to train. Supported values: 'CatBoost', 'RF'
    tuning_results : Dict[str, Any]
        Dictionary from tune_hyperparameters containing:
        - 'best_params': Dict[str, Any], optimized hyperparameters
        - 'cv_results': Dict[str, Any], cross-validation results (optional, will be
          included in return if present)

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'model': Trained model instance (CatBoostRegressor or RandomForestRegressor)
        - 'best_params': Dict[str, Any], copy of the best_params used for training
        - 'x_scaler': StandardScaler, fitted scaler used to transform features
        - 'input_example': pd.DataFrame, first 5 rows of x_train (used for MLflow
          signature inference)
        - 'cv_results': Dict[str, Any], cross-validation results from best model
          (if present in tuning_results)
    """

    model_name = params["optuna_search"]["model"]

    x_scaler = StandardScaler()
    x_scaled = x_scaler.fit_transform(x_train)

    if model_name == "CatBoost":
        model = CatBoostRegressor(
            **tuning_results["best_params"],
            allow_writing_files=False,
            verbose=False,
        )
    elif model_name == "RF":
        model = RF(**tuning_results["best_params"])
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    model.fit(x_scaled, y_train)

    y_pred_test = model.predict(x_scaler.transform(x_test))
    errors = compute_metrics(y_test, y_pred_test)

    return {
        "model": model,
        "x_scaler": x_scaler,
        "input_example": x_train.iloc[:5].copy(),
        "test_metrics": {
            "test_mae": errors["mae"],
            "test_rmse": errors["rmse"],
            "test_mape": errors["mape"],
        },
    }