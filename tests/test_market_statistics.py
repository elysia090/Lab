from __future__ import annotations

from pathlib import Path
import sys

import pytest

np = pytest.importorskip("numpy")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from finance.market_simulation.market_model import (  # type: ignore  # pylint: disable=import-error
    compute_equity_statistics,
    format_statistics_table,
    summarise_equity_curves,
)


def test_compute_equity_statistics_known_values():
    equity_curve = np.array([100.0, 110.0, 105.0, 115.0])
    stats = compute_equity_statistics(
        equity_curve,
        periods_per_year=4,
        risk_free_rate=0.02,
    )

    assert stats.final_equity == pytest.approx(115.0)
    assert stats.total_return == pytest.approx(0.15)
    assert stats.cagr == pytest.approx(0.2048429861)
    assert stats.volatility == pytest.approx(0.1652757689)
    assert stats.max_drawdown == pytest.approx(0.04545454545)
    assert stats.sharpe_ratio == pytest.approx(1.0873426934)


def test_compute_equity_statistics_handles_flat_curve():
    equity_curve = np.array([100.0, 100.0, 100.0])
    stats = compute_equity_statistics(equity_curve)

    assert stats.total_return == pytest.approx(0.0)
    assert stats.cagr == pytest.approx(0.0)
    assert stats.volatility == pytest.approx(0.0)
    assert stats.max_drawdown == pytest.approx(0.0)
    assert stats.sharpe_ratio == pytest.approx(0.0)


def test_format_statistics_table_rounds_values():
    equity_curves = [
        np.array([100.0, 110.0, 105.0, 115.0]),
        np.array([100.0, 95.0, 97.0, 96.0]),
    ]
    stats = summarise_equity_curves(equity_curves, periods_per_year=4)
    table = format_statistics_table(stats)

    assert "Simulation 1" in table
    assert "Simulation 2" in table
    assert "15.00%" in table  # total return of the first simulation
    assert "%" in table  # ensure percentage formatting is applied
