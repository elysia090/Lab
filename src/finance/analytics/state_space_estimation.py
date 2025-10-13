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
from pathlib import Path
from typing import Iterable, Sequence

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

    def state_transition(self, state: np.ndarray) -> tuple[np.ndarray, float]:
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

        if state.ndim != 1:
            raise ValueError("State vector must be one-dimensional.")
        if state.size != self.state_dimension:
            raise ValueError(
                "State vector has incorrect dimension: "
                f"expected {self.state_dimension}, received {state.size}."
            )

        ar_contribution = 0.0
        if self.ar_params.size:
            ar_contribution = float(
                np.dot(self.ar_params, state[-self.ar_params.size :])
            )

        ma_contribution = 0.0
        if self.ma_params.size:
            ma_contribution = float(
                np.dot(self.ma_params, state[-self.ma_params.size :])
            )

        predicted_observation = ar_contribution + ma_contribution
        next_state = np.roll(state, -1)
        next_state[-1] = predicted_observation
        return next_state, predicted_observation


class ExtendedKalmanFilter:
    """Minimal Extended Kalman Filter operating on :class:`StateSpaceModel`."""

    def __init__(self, state_space_model: StateSpaceModel, process_noise_cov: np.ndarray) -> None:
        self.state_space_model = state_space_model
        self.process_noise_cov = np.asarray(process_noise_cov, dtype=float)
        self.state_estimation: np.ndarray | None = None
        self.state_covariance: np.ndarray | None = None

    def initialize(self, initial_state: np.ndarray, initial_state_cov: np.ndarray) -> None:
        state = np.asarray(initial_state, dtype=float)
        covariance = np.asarray(initial_state_cov, dtype=float)

        if state.ndim != 1:
            raise ValueError("Initial state must be a one-dimensional vector.")
        if state.size != self.state_space_model.state_dimension:
            raise ValueError(
                "Initial state does not match the state-space model dimension."
            )
        if covariance.shape != (state.size, state.size):
            raise ValueError(
                "Initial covariance matrix must be square with size matching the state vector."
            )
        if self.process_noise_cov.shape != covariance.shape:
            raise ValueError(
                "Process noise covariance must match the shape of the covariance matrix."
            )

        self.state_estimation = state.copy()
        self.state_covariance = covariance.copy()

    def _ensure_initialized(self) -> None:
        if self.state_estimation is None or self.state_covariance is None:
            raise RuntimeError("Kalman filter has not been initialised yet.")

    def predict(self, control_signal: float = 0.0) -> float:
        """Predict the next observation given an optional control signal."""

        self._ensure_initialized()
        assert self.state_estimation is not None  # for mypy
        assert self.state_covariance is not None

        next_state, predicted_observation = self.state_space_model.state_transition(
            self.state_estimation
        )
        next_state[-1] += float(control_signal)
        self.state_estimation = next_state
        self.state_covariance = self.state_covariance + self.process_noise_cov
        return float(self.state_estimation[-1])

    def update(self, observation: float) -> float:
        """Correct the prediction using the provided observation."""

        self._ensure_initialized()
        assert self.state_estimation is not None
        assert self.state_covariance is not None

        measurement_vector = np.zeros_like(self.state_estimation)
        measurement_vector[-1] = 1.0

        predicted_observation = float(self.state_estimation[-1])
        innovation = float(observation) - predicted_observation

        s = float(
            measurement_vector @ self.state_covariance @ measurement_vector.T
        ) + float(self.process_noise_cov[-1, -1])
        if s <= 0:
            raise RuntimeError("Innovation covariance must be positive.")

        kalman_gain = (self.state_covariance @ measurement_vector) / s
        self.state_estimation = self.state_estimation + kalman_gain * innovation
        identity = np.eye(self.state_covariance.shape[0])
        self.state_covariance = (
            identity - np.outer(kalman_gain, measurement_vector)
        ) @ self.state_covariance
        return float(self.state_estimation[-1])


class CWLEM:
    """Clifford Wide-Area Linear Estimation Method approximation.

    In practice this simply implements a normalised weighted sum over external
    factors.  The class performs validation and gracefully falls back to a
    uniform weighting scheme when the provided weights are ill-conditioned.
    """

    def __init__(self, weights: Iterable[float]) -> None:
        weights_array = np.asarray(list(weights), dtype=float)
        if weights_array.ndim != 1 or not weights_array.size:
            raise ValueError("Weights must be a one-dimensional, non-empty iterable.")

        total = float(np.sum(weights_array))
        if not np.isfinite(total) or abs(total) < np.finfo(float).eps:
            weights_array = np.ones_like(weights_array)
            total = float(weights_array.size)

        self.weights = weights_array / total

    def predict(self, external_factors: Sequence[float]) -> float:
        factors_array = np.asarray(external_factors, dtype=float)
        if factors_array.ndim != 1:
            raise ValueError("External factors must be one-dimensional.")
        if factors_array.size != self.weights.size:
            raise ValueError(
                "Number of external factors does not match the number of weights."
            )
        return float(np.dot(factors_array, self.weights))


class PerformanceEvaluator:
    """Collection of evaluation metrics for the forecasting pipeline."""

    @staticmethod
    def calculate_rmsle(predictions: np.ndarray, true_values: np.ndarray) -> float:
        predictions = np.asarray(predictions, dtype=float)
        true_values = np.asarray(true_values, dtype=float)

        if predictions.shape != true_values.shape:
            raise ValueError("Predictions and true values must share the same shape.")
        if np.any(predictions < 0) or np.any(true_values < 0):
            raise ValueError("RMSLE is undefined for negative values.")

        log_diff = np.log1p(predictions) - np.log1p(true_values)
        return float(np.sqrt(np.mean(np.square(log_diff))))


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

    data = pd.read_csv(path, usecols=list(columns))
    if data.empty:
        raise ValueError(f"File is empty or missing requested columns: {path}")
    return data


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
    weights = correlation.mean(axis=0).to_numpy(dtype=float)

    if not np.isfinite(weights).all() or np.allclose(weights, 0):
        weights = np.ones(external_factors.shape[1], dtype=float)

    weights /= np.sum(weights)
    return weights


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

    model = ARIMA(
        train_series.to_numpy(dtype=float),
        order=arima_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = model.fit()

    state_space_model = StateSpaceModel(
        ar_params=results.arparams if results.arparams is not None else np.array([]),
        ma_params=results.maparams if results.maparams is not None else np.array([]),
    )

    state_dimension = state_space_model.state_dimension
    initial_state = np.zeros(state_dimension, dtype=float)
    covariance_scale = float(process_noise)
    initial_state_cov = np.eye(state_dimension, dtype=float) * covariance_scale
    process_noise_cov = np.eye(state_dimension, dtype=float) * covariance_scale

    kalman_filter = ExtendedKalmanFilter(state_space_model, process_noise_cov)
    kalman_filter.initialize(initial_state, initial_state_cov)

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

