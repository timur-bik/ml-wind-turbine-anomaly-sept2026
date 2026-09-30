import pandas as pd
import numpy as np

def rename_columns(df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """
    Rename columns in a dataframe.
    """
    return df.rename(columns=columns)


def drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Drop columns in a dataframe.
    """
    return df.drop(columns=columns)


def remove_diff_outliers(df: pd.DataFrame, diff_thresholds: dict[str, float]) -> pd.DataFrame:
    """
    Remove outliers based on absolute first-order diff and forward-fill the gaps.
    Parameters:
    ----------
    df: pd.DataFrame
        Input dataframe.
    diff_thresholds: dict[str, float]
        Dictionary of column names and their diff thresholds.
    """
    df_clean = df.copy()
    for col, threshold in diff_thresholds.items():
        # 1. Compute absolute diff
        diff_vals = df_clean[col].diff(1).abs()

        # 2. Outlier mask
        outlier_mask = diff_vals > threshold
        outlier_idx = df_clean.index[outlier_mask]

        # 3. Remove outliers
        df_clean.loc[outlier_idx, col] = np.nan

        # 4. Forward fill (and backfill if needed)
        df_clean[col] = df_clean[col].ffill().bfill()

    return df_clean


def smooth_signal(df: pd.DataFrame, columns: str, window: int, method: str = "mean") -> pd.DataFrame:
    """
    Smooth signal in a dataframe.
    Parameters:
    ----------
    df: pd.DataFrame
        Input dataframe.
    columns: str
        Columns to smooth.
    window: int
        Window size for smoothing.
    """
    df_smoothed = df.copy()

    for col in columns:
        if method == "mean":
            df_smoothed[col] = df_smoothed[col].rolling(
                window=window, min_periods=1, center=False
            ).mean()

        elif method == "median":
            df_smoothed[col] = df_smoothed[col].rolling(
                window=window, min_periods=1, center=False
            ).median()

        else:
            raise ValueError("method must be 'mean' or 'median'")

    return df_smoothed


def add_lag_features(df: pd.DataFrame, lags_dict: dict[str, list[int]], drop_na=False) -> pd.DataFrame:
    """
    Add lag features to a dataframe.
    Parameters:
    ----------
    df: pd.DataFrame
        Input dataframe.
    lags_dict: dict[str, list[int]]
        Dictionary of column names and their lags.
    drop_na: bool
        Whether to drop NaNs.
    """
    df_result = df.copy()

    # Create lag features
    for col, lags in lags_dict.items():
        for lag in lags:
            df_result[f"{col}_lag{lag}"] = df_result[col].shift(lag)

    if drop_na:
        return df_result.dropna()
    else:
        return df_result.bfill()  # Backward fill NaNs


def add_rolling_features(
    df: pd.DataFrame,
    stats_window_dict: dict[str, dict[str, list]],
    drop_na=False,
) -> pd.DataFrame:
    """
    Add rolling features to a dataframe.
    Parameters:
    ----------
    df: pd.DataFrame
        Input dataframe.
    stats_window_dict: dict[str, dict[str, list]]
        Dictionary of column names and their rolling features.
    drop_na: bool
        Whether to drop NaNs.
    """
    df_result = df.copy()

    for col in stats_window_dict.keys():
        for window in stats_window_dict[col]["windows"]:
            rolling = df_result[col].rolling(window)
            for stat in stats_window_dict[col]["stats"]:
                if stat == "mean":
                    df_result[f"{col}_roll{window}_mean"] = rolling.mean()
                elif stat == "median":
                    df_result[f"{col}_roll{window}_median"] = rolling.median()
                elif stat == "std":
                    df_result[f"{col}_roll{window}_std"] = rolling.std()
                elif stat == "min":
                    df_result[f"{col}_roll{window}_min"] = rolling.min()
                elif stat == "max":
                    df_result[f"{col}_roll{window}_max"] = rolling.max()
                elif stat == "skew":
                    df_result[f"{col}_roll{window}_skew"] = rolling.skew()
                elif stat == "kurt":
                    df_result[f"{col}_roll{window}_kurt"] = rolling.kurt()
    if drop_na:
        return df_result.dropna()
    else:
        return df_result.bfill()  # Backward fill NaNs


def get_features_and_target(
    df: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get the features and target from the dataframe.
    """
    x = df.drop(columns=target).copy()
    y = df[[target]].copy()
    return x, y