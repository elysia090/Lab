from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("statsmodels")

from finance.analytics.state_space_estimation import (
    PerformanceEvaluator,
    forecast_sales_with_kalman,
    _prepare_external_factors,
)


def test_forecast_sales_with_kalman_predictions_shape() -> None:
    rng = np.random.default_rng(0)
    periods = 40
    index = pd.date_range("2023-01-01", periods=periods, freq="D")

    trend = np.linspace(100, 120, periods)
    seasonal = 5 * np.sin(np.linspace(0, 2 * np.pi, periods))
    noise = rng.normal(scale=0.3, size=periods)
    sales = pd.Series(trend + seasonal + noise + 50, index=index)

    external_factors = pd.DataFrame(
        {
            "oil": np.linspace(40, 60, periods) + rng.normal(scale=0.2, size=periods),
            "transactions": np.linspace(200, 260, periods)
            + rng.normal(scale=0.5, size=periods),
        },
        index=index,
    )

    result = forecast_sales_with_kalman(sales, external_factors)

    assert len(result.predictions) == periods
    assert np.isfinite(result.predictions).all()
    assert result.rmsle >= 0


def test_forecast_sales_with_kalman_rejects_negative_sales() -> None:
    sales = pd.Series([10.0, -1.0, 12.0])
    factors = pd.DataFrame({"factor": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="non-negative"):
        forecast_sales_with_kalman(sales, factors)


def test_prepare_external_factors_length_mismatch() -> None:
    factors = [
        pd.DataFrame({"date": [1, 2, 3], "value": [0.1, 0.2, 0.3]}),
        pd.DataFrame({"date": [1, 2], "value": [0.4, 0.5]}),
    ]

    with pytest.raises(ValueError, match="length mismatch"):
        _prepare_external_factors(factors, expected_length=3)


def test_performance_evaluator_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="negative"):
        PerformanceEvaluator.calculate_rmsle(np.array([1.0, -1.0]), np.array([1.0, 1.0]))
