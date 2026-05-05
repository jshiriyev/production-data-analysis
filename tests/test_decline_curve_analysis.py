# tests/test_dca.py
import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from numpy.testing import assert_allclose

from prodpy import DCA

try:
    from prodpy.decline import Arps
except Exception:  # pragma: no cover - fallback for local runs
    from prodpy.decline._arps import Arps


@pytest.fixture
def synthetic_decline_df():
    dates = pd.date_range("2022-01-01", periods=30, freq="D")
    t = np.arange(dates.size, dtype=float)
    base = Arps(di=0.25, qi=150.0, b=0.5)
    rates = base.run(t, cum=False)
    return pd.DataFrame({"date": dates, "rate": rates})


def test_dca_fit_and_run_produces_expected_grid(synthetic_decline_df):
    dca = DCA(synthetic_decline_df)
    result = dca.fit(model="hyperbolic", b=0.5).run(horizon_days=60)

    expected_columns = ["date", "t_days", "q_hist", "q_fit", "q_forecast", "N_forecast"]
    assert list(result.columns) == expected_columns

    hist_len = synthetic_decline_df.shape[0]
    assert result.shape[0] > hist_len

    hist_rates = result["q_hist"].iloc[:hist_len].to_numpy()
    assert_allclose(hist_rates, synthetic_decline_df["rate"].to_numpy(), rtol=1e-12, atol=1e-12)
    assert result["q_hist"].isna().sum() == result.shape[0] - hist_len

    if hist_len > 1:
        assert result["q_forecast"].iloc[: hist_len - 1].isna().all()


def test_dca_run_applies_economic_cutoff(synthetic_decline_df):
    dca = DCA(synthetic_decline_df).fit(model="hyperbolic", b=0.5)
    econ_rate = 5.0

    baseline = dca.run(horizon_days=365)
    q_baseline = baseline["q_fit"].to_numpy()
    below = np.nonzero(q_baseline < econ_rate)[0]
    assert below.size > 0

    cut_idx = int(below[0])
    assert cut_idx >= synthetic_decline_df.shape[0]

    result = dca.run(horizon_days=365, econ_rate=econ_rate)

    q_cut = result["q_fit"].to_numpy()
    n_cut = result["N_forecast"].to_numpy()
    assert np.isnan(q_cut[cut_idx:]).all()
    assert np.isnan(n_cut[cut_idx:]).all()


def test_dca_plot_returns_axes(synthetic_decline_df):
    dca = DCA(synthetic_decline_df).fit(model="hyperbolic", b=0.5)
    ax = dca.plot(show=False, horizon_days=90)

    assert isinstance(ax, Axes)
    plt.close(ax.figure)
