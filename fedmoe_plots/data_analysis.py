"""Utilities for data analysis in the FedMoE project."""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

STEP_COLUMN = "time/batch"
TOKEN_COUNT_COLUMN = "time/token"  # noqa: S105
DEVICE_THROUGHPUT_TOKENS = "throughput/device/tokens_per_sec"
TRAIN_PERPLEXITY_COLUMN = "metrics/train/LanguagePerplexity"


class ColumnNotFoundError(Exception):
    """Exception raised when a required column is not found in the DataFrame."""

    def __init__(self, column_name: str) -> None:
        """Initialize the exception with the column name.

        Parameters
        ----------
        column_name : str
            The name of the column that was not found.

        """
        super().__init__(f"Column '{column_name}' not found in DataFrame.")
        self.column_name = column_name


def get_smoothed_series(
    client_metrics_df: pd.DataFrame,
    metric_name: str = TRAIN_PERPLEXITY_COLUMN,
    *,
    moving_window: int = 5,
    linear_interpolation: bool = True,
) -> tuple[pd.Series, pd.Series]:
    """Get a smoothed series of a specific metric from the client metrics DataFrame.

    Parameters
    ----------
    client_metrics_df : pd.DataFrame
        The DataFrame containing client metrics.
    metric_name : str, optional
        The name of the metric to extract, by default
        TRAIN_PERPLEXITY_COLUMN.
    moving_window : int, optional
        The size of the moving window for smoothing, by default 5.
    linear_interpolation : bool, optional
        Whether to apply linear interpolation to fill NaN values, by default True.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        A tuple containing two pandas Series:
        - The first Series contains the steps.
        - The second Series contains the smoothed metric values.

    Raises
    ------
    ColumnNotFoundError
        If the specified metric column is not found in the DataFrame.

    """
    # Directly select only columns of interest
    columns_of_interest = [STEP_COLUMN, metric_name]
    for col in columns_of_interest:
        if col not in client_metrics_df.columns:
            raise ColumnNotFoundError(col)
    filtered_df = client_metrics_df[columns_of_interest].copy()

    # Remove inf values and substitute non-numeric values with NaN
    filtered_df = filtered_df.replace([np.inf, -np.inf], np.nan)
    filtered_df = filtered_df.apply(pd.to_numeric, errors="coerce")

    # Aggregate by step and compute mean
    aggregated_df = filtered_df.groupby(STEP_COLUMN).min().reset_index()

    # Fill NaNs by interpolation
    if linear_interpolation:
        aggregated_df = aggregated_df.interpolate(method="linear")

    # Extract series
    steps = aggregated_df[STEP_COLUMN]
    perplexity_series = aggregated_df[metric_name]

    # Smooth the perplexity series
    smoothed_perplexity = perplexity_series.rolling(window=moving_window).mean()

    return steps, smoothed_perplexity


def get_global_token_series(
    client_metrics_df: pd.DataFrame,
    n_clients_per_round: int,
) -> tuple[pd.Series, pd.Series]:
    """Get the global token series from the client metrics DataFrame.

    Parameters
    ----------
    client_metrics_df : pd.DataFrame
        The DataFrame containing client metrics.
    n_clients_per_round : int
        The number of clients per round.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        A tuple containing two pandas Series:
        - The first Series contains the steps.
        - The second Series contains the global token count.

    """
    # Get the original series
    steps, tokens = get_smoothed_series(
        client_metrics_df,
        metric_name=TOKEN_COUNT_COLUMN,
        moving_window=1,
    )
    # Multiply by the number of clients per round
    tokens *= n_clients_per_round

    return steps, tokens


def get_device_throughput_series(
    client_metrics_df: pd.DataFrame,
    moving_window: int,
) -> tuple[pd.Series, pd.Series]:
    """Get the device throughput series from the client metrics DataFrame.

    Parameters
    ----------
    client_metrics_df : pd.DataFrame
        The DataFrame containing client metrics.
    moving_window : int
        The size of the moving window for smoothing.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        A tuple containing two pandas Series:
        - The first Series contains the steps.
        - The second Series contains the device throughput in tokens per second.

    """
    # Get the original series
    steps, tokens = get_smoothed_series(
        client_metrics_df,
        metric_name=DEVICE_THROUGHPUT_TOKENS,
        moving_window=moving_window,
    )
    # exclude NaN values
    valid_mask = ~tokens.isna()
    steps = steps[valid_mask]
    tokens = tokens[valid_mask]

    return steps, tokens


def get_perplexity_versus_tokens(
    client_metrics_df: pd.DataFrame,
    n_clients_per_round: int,
    moving_window: int = 5,
) -> tuple[pd.Series, pd.Series]:
    """Get the perplexity versus tokens series from the client metrics DataFrame.

    Parameters
    ----------
    client_metrics_df : pd.DataFrame
        The DataFrame containing client metrics.
    n_clients_per_round : int
        The number of clients per round.
    moving_window : int, optional
        The size of the moving window for smoothing, by default 5.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        A tuple containing two pandas Series:
        - The first Series contains the global token count.
        - The second Series contains the smoothed perplexity values.

    Raises
    ------
    ValueError
        If the tokens and perplexity series do not have the same length.

    """
    _, tokens = get_global_token_series(client_metrics_df, n_clients_per_round)
    _, perplexity = get_smoothed_series(
        client_metrics_df,
        metric_name=TRAIN_PERPLEXITY_COLUMN,
        moving_window=moving_window,
    )
    # Ensure tokens and perplexity are aligned
    if len(tokens) != len(perplexity):
        msg = "Tokens and perplexity series must have the same length."
        raise ValueError(msg)

    return tokens, perplexity
