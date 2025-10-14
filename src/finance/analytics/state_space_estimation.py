"""State space estimation utilities built around a lightweight Kalman workflow.

The original implementation of this module focused on loading a Kaggle
competition dataset and running a loosely defined Extended Kalman Filter loop.
It relied heavily on implicit assumptions (for example that every external
factor lived in the second column of a CSV) and silently swallowed many error
conditions.  The refactor performed in this commit focuses on three goals:

* Provide a well documented, testable core API that can be reused by the rest
  of the repository.
* Enforce strong validation with informative exceptions instead of printing
  errors to stdout.
* Keep the numerical routines small and dependency-light while still exposing
  hooks for ARIMA based initialisation.

The public entry point is :func:`forecast_sales_with_kalman`, which accepts a
``pandas.Series`` containing the target variable (typically sales) together with
tabular external factors.  The function orchestrates the ARIMA estimation of the
autoregressive parameters, sets up a small Extended Kalman Filter and returns a
``ForecastResult`` describing the predictions together with the RMSLE metric.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Iterable, Sequence

import math

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

__all__ = [
    "CWLEM",
    "ExtendedKalmanFilter",
    "ForecastResult",
    "PerformanceEvaluator",
    "StateSpaceModel",
    "forecast_sales_with_kalman",
    "initialize_models",
    "load_csv_columns",
]


@dataclass
class StateSpaceModel:
    """Compact ARMA style state transition model.

    The model keeps track of a one dimensional hidden state that stores the
    most recent autoregressive (AR) and moving average (MA) contributions.  The
    length of the state is determined by the larger of ``len(ar_params)`` or
    ``len(ma_params)`` with a minimum size of one to keep the matrix algebra
    well-defined.
    """

    ar_params: np.ndarray
    ma_params: np.ndarray

    def __post_init__(self) -> None:
        self.ar_params = np.asarray(self.ar_params, dtype=float)
        self.ma_params = np.asarray(self.ma_params, dtype=float)

    @property
    def state_dimension(self) -> int:
        """Number of elements that need to be tracked in the hidden state."""

        return max(int(self.ar_params.size), int(self.ma_params.size), 1)

    def state_transition(self, state: Sequence[float]) -> tuple[list[float], float]:
        """Advance the state by one step.

        Parameters
        ----------
        state:
            Current hidden state vector.

        Returns
        -------
        tuple[np.ndarray, float]
            The updated state vector and the predicted observation before any
            external control signal is applied.
        """

        if len(state) != self.state_dimension:
            raise ValueError(
                "State vector has incorrect dimension: "
                f"expected {self.state_dimension}, received {len(state)}."
            )

        ar_order = len(self.ar_params)
        ma_order = len(self.ma_params)

        ar_window = state[-ar_order:] if ar_order else []
        ma_window = state[-ma_order:] if ma_order else []

        ar_contribution = float(np.dot(self.ar_params, ar_window)) if ar_window else 0.0
        ma_contribution = float(np.dot(self.ma_params, ma_window)) if ma_window else 0.0

        predicted_observation = ar_contribution + ma_contribution
        next_state = list(state[1:]) + [predicted_observation]
        return next_state, predicted_observation


class ExtendedKalmanFilter:
    """Tiny scalar Kalman filter used for the compatibility implementation."""

    def __init__(self, state_space_model: StateSpaceModel, process_noise: float) -> None:
        self.state_space_model = state_space_model
        self.process_noise = float(process_noise)
        self.state_estimation: list[float] | None = None
        self.state_variance: float | None = None

    def initialize(self, initial_state: Sequence[float], initial_variance: float) -> None:
        state_list = [float(value) for value in initial_state]
        if len(state_list) != self.state_space_model.state_dimension:
            raise ValueError(
                "Initial state does not match the state-space model dimension."
            )
        self.state_estimation = state_list
        self.state_variance = float(initial_variance)

    def _ensure_initialized(self) -> None:
        if self.state_estimation is None or self.state_variance is None:
            raise RuntimeError("Kalman filter has not been initialised yet.")

    def predict(self, control_signal: float = 0.0) -> float:
        self._ensure_initialized()
        assert self.state_estimation is not None
        assert self.state_variance is not None

        next_state, predicted = self.state_space_model.state_transition(
            self.state_estimation
        )
        next_state[-1] += float(control_signal)
        self.state_estimation = next_state
        self.state_variance = self.state_variance + self.process_noise
        return float(self.state_estimation[-1])

    def update(self, observation: float) -> float:
        self._ensure_initialized()
        assert self.state_estimation is not None
        assert self.state_variance is not None

        predicted = float(self.state_estimation[-1])
        innovation = float(observation) - predicted
        innovation_covariance = self.state_variance + self.process_noise
        if innovation_covariance <= 0:
            raise RuntimeError("Innovation covariance must be positive.")

        kalman_gain = self.state_variance / innovation_covariance
        self.state_estimation[-1] = predicted + kalman_gain * innovation
        self.state_variance = (1.0 - kalman_gain) * self.state_variance
        return float(self.state_estimation[-1])


class CWLEM:
    """Clifford Wide-Area Linear Estimation Method approximation.

    In practice this simply implements a normalised weighted sum over external
    factors.  The class performs validation and gracefully falls back to a
    uniform weighting scheme when the provided weights are ill-conditioned.
    """

    def __init__(self, weights: Iterable[float]) -> None:
        values = [float(weight) for weight in weights]
        if not values:
            raise ValueError("Weights must be a non-empty iterable.")

        total = sum(values)
        if not np.isfinite([total]).all() or abs(total) < 1e-12:
            values = [1.0 for _ in values]
            total = float(len(values))

        self.weights = [value / total for value in values]

    def predict(self, external_factors: Sequence[float]) -> float:
        factors = [float(value) for value in external_factors]
        if len(factors) != len(self.weights):
            raise ValueError(
                "Number of external factors does not match the number of weights."
            )
        return float(sum(factor * weight for factor, weight in zip(factors, self.weights)))


class PerformanceEvaluator:
    """Collection of evaluation metrics for the forecasting pipeline."""

    @staticmethod
    def calculate_rmsle(predictions: Sequence[float], true_values: Sequence[float]) -> float:
        pred = [float(value) for value in predictions]
        truth = [float(value) for value in true_values]

        if len(pred) != len(truth):
            raise ValueError("Predictions and true values must share the same shape.")
        if any(value < 0 for value in pred) or any(value < 0 for value in truth):
            raise ValueError("RMSLE is undefined for negative values.")

        log_diff = [math.log1p(p) - math.log1p(t) for p, t in zip(pred, truth)]
        mean_square = np.mean(np.square(log_diff))
        return float(np.sqrt(mean_square))


@dataclass(frozen=True)
class ForecastResult:
    """Container returned by :func:`forecast_sales_with_kalman`."""

    predictions: np.ndarray
    rmsle: float


def load_csv_columns(file_path: str | Path, columns: Sequence[str]) -> pd.DataFrame:
    """Load a subset of columns from ``file_path``.

    Parameters
    ----------
    file_path:
        Location of the CSV file.
    columns:
        Iterable describing the columns that should be loaded.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the requested columns.
    """

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    requested = list(columns)
    rows: dict[str, list[float]] = {column: [] for column in requested}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"File is empty or missing requested columns: {path}")
        missing = [column for column in requested if column not in reader.fieldnames]
        if missing:
            raise ValueError(
                "File is empty or missing requested columns: "
                f"{path} (missing {', '.join(missing)})"
            )
        for row in reader:
            for column in requested:
                rows[column].append(float(row[column]))
    if not any(rows.values()):
        raise ValueError(f"File is empty or missing requested columns: {path}")
    return pd.DataFrame(rows)


def _prepare_external_factors(
    obs_data: Sequence[pd.DataFrame] | pd.DataFrame,
    expected_length: int,
) -> pd.DataFrame:
    """Normalise the representation of external factor data.

    The historical code assumed that every observation matrix contained the
    desired numeric feature in the second column.  The helper now accepts either
    a fully formed :class:`pandas.DataFrame` or a sequence of frames and returns
    a clean numeric matrix with aligned indices.
    """

    if isinstance(obs_data, pd.DataFrame):
        factors = obs_data.copy()
    else:
        series_collection: list[pd.Series] = []
        for index, frame in enumerate(obs_data):
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(
                    "Each element of obs_data must be a pandas.DataFrame."
                )
            if len(frame) != expected_length:
                raise ValueError(
                    "Observation data length mismatch: "
                    f"expected {expected_length}, received {len(frame)} for index {index}."
                )
            numeric_series = pd.to_numeric(frame.iloc[:, -1], errors="coerce")
            if numeric_series.isna().any():
                raise ValueError(
                    "Observation data contains non-numeric values that cannot be coerced."
                )
            series_collection.append(numeric_series.reset_index(drop=True))

        factors = pd.concat(series_collection, axis=1)
        factors.columns = [f"factor_{i}" for i in range(len(series_collection))]

    if len(factors) != expected_length:
        raise ValueError(
            "External factor rows do not align with the target series length."
        )

    numeric_factors = factors.apply(pd.to_numeric, errors="raise").astype(float)
    if numeric_factors.isnull().any().any():
        raise ValueError("External factor matrix contains NaN entries.")
    return numeric_factors.reset_index(drop=True)


def _compute_feature_weights(external_factors: pd.DataFrame) -> np.ndarray:
    """Derive stable weights for :class:`CWLEM` from correlation structure."""

    if external_factors.empty:
        raise ValueError("At least one external factor is required.")

    correlation = external_factors.corr().abs().fillna(0.0)
    weights_series = correlation.mean(axis=0)
    weights = list(weights_series.to_numpy(dtype=float))

    if not np.isfinite(weights).all() or all(abs(value) < 1e-12 for value in weights):
        weights = [1.0] * len(weights)

    total = sum(weights)
    return np.asarray([value / total for value in weights])


def initialize_models(
    train_series: pd.Series,
    external_factors: pd.DataFrame,
    *,
    arima_order: tuple[int, int, int] = (1, 1, 1),
    process_noise: float = 1.0,
) -> tuple[ExtendedKalmanFilter, CWLEM]:
    """Initialise the Kalman filter and CWLEM helper.

    Parameters
    ----------
    train_series:
        Target time series to model.  The index is ignored; only the values are
        used.
    external_factors:
        DataFrame containing aligned external factors.
    arima_order:
        Order used when fitting the ARIMA model to bootstrap the AR and MA
        parameters.
    process_noise:
        Scalar used to scale the diagonal process noise covariance matrix.
    """

    if train_series.empty:
        raise ValueError("Training series must not be empty.")

    model = ARIMA(train_series.to_numpy(dtype=float), order=arima_order)
    results = model.fit()

    state_space_model = StateSpaceModel(
        ar_params=results.arparams if results.arparams is not None else np.array([]),
        ma_params=results.maparams if results.maparams is not None else np.array([]),
    )

    state_dimension = state_space_model.state_dimension
    initial_state = [0.0 for _ in range(state_dimension)]
    kalman_filter = ExtendedKalmanFilter(state_space_model, float(process_noise))
    kalman_filter.initialize(initial_state, float(process_noise))

    weights = _compute_feature_weights(external_factors)
    cwlem = CWLEM(weights)
    return kalman_filter, cwlem


def forecast_sales_with_kalman(
    sales: pd.Series,
    external_factors: Sequence[pd.DataFrame] | pd.DataFrame,
    *,
    arima_order: tuple[int, int, int] = (1, 1, 1),
    process_noise: float = 1.0,
) -> ForecastResult:
    """Run the full forecasting pipeline.

    Parameters
    ----------
    sales:
        Target series that should be forecast.  The values must be non-negative
        to keep the RMSLE metric well-defined.
    external_factors:
        Either a DataFrame containing aligned factors or a sequence of
        DataFrames as produced by the original scripts.
    arima_order, process_noise:
        Parameters forwarded to :func:`initialize_models`.
    """

    sales_values = sales.to_numpy(dtype=float)
    if np.any(~np.isfinite(sales_values)):
        raise ValueError("Sales series contains NaN or infinite values.")
    if np.any(sales_values < 0):
        raise ValueError("Sales series must be non-negative for RMSLE.")

    factors = _prepare_external_factors(external_factors, len(sales))
    kalman_filter, cwlem = initialize_models(
        sales, factors, arima_order=arima_order, process_noise=process_noise
    )

    predictions: list[float] = []
    factors_array = factors.to_numpy(dtype=float)
    for observation, factor_row in zip(sales_values, factors_array, strict=True):
        control_signal = cwlem.predict(factor_row)
        predicted_value = kalman_filter.predict(control_signal)
        predictions.append(predicted_value)
        kalman_filter.update(observation)

    predictions_array = np.asarray(predictions, dtype=float)
    safe_predictions = np.clip(predictions_array, a_min=0.0, a_max=None)
    rmsle = PerformanceEvaluator.calculate_rmsle(safe_predictions, sales_values)
    return ForecastResult(predictions=predictions_array, rmsle=rmsle)


__all__ = sorted(__all__)

