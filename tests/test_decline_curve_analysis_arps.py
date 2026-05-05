# tests/test_arps.py
import math
import numpy as np
import pytest
from numpy.testing import assert_allclose

# Adjust import path to your project
from prodpy.decline import Arps
from prodpy.decline import Exponential
from prodpy.decline import Harmonic
from prodpy.decline import Hyperbolic

@pytest.fixture(params=["exponential", "harmonic", "hyperbolic"])
def kind(request):
    return request.param

@pytest.fixture
def params(kind):
    if kind == "exponential":
        return dict(mode="exponential", di=0.25, qi=120.0, b=0.0)
    if kind == "harmonic":
        return dict(mode="harmonic", di=0.25, qi=120.0, b=1.0)
    # hyperbolic
    return dict(mode="hyperbolic", di=0.25, qi=120.0, b=0.5)

@pytest.fixture
def tvec():
    return np.linspace(0.0, 10.0, 1201)  # dense for robust tests

def _model_from_params(params):
    mode = params["mode"]
    di = params["di"]
    qi = params["qi"]
    b = params["b"]
    if mode == "exponential":
        return Exponential(di=di, qi=qi)
    if mode == "harmonic":
        return Harmonic(di=di, qi=qi)
    return Hyperbolic(b=b, di=di, qi=qi)

# -----------------------
# Construction & mapping
# -----------------------
def test_mode_b_mapping(params):
    mode = params["mode"]
    b = params["b"]
    m = Arps(params["di"], params["qi"], b=b)
    assert m.mode == mode
    assert math.isclose(m.b, b, rel_tol=0, abs_tol=0)

    # reverse mapping
    mode2, b2 = Arps.option(mode=mode, b=None)
    assert mode2.lower() == mode
    assert math.isclose(b2, Arps.mode2b(mode), rel_tol=0, abs_tol=0)

# --------------
# run() behavior
# --------------
@pytest.mark.parametrize("cum", [False, True])
def test_run_matches_underlying(params, tvec, cum):
    base = _model_from_params(params)
    arps = Arps(params["di"], params["qi"], b=params["b"])

    y_ref = base.N(tvec) if cum else base.q(tvec)
    y = arps.run(tvec, cum=cum)
    assert_allclose(y, y_ref, rtol=1e-12, atol=1e-12)

def test_run_with_xi_shift(params, tvec):
    base = _model_from_params(params)
    arps = Arps(params["di"], params["qi"], b=params["b"])

    xi = 2.0
    y = arps.run(tvec, xi=xi, cum=False)
    # for t < xi we expect NaN (we masked in the implementation)
    assert np.isnan(y[tvec < xi]).all()
    # for t >= xi, this equals q(t - xi)
    mask = tvec >= xi
    assert_allclose(y[mask], base.q(tvec[mask] - xi), rtol=1e-12, atol=1e-12)

# -----------------------------
# linregress / fit per model
# -----------------------------
@pytest.mark.parametrize("noise_level", [0.0, 0.002])
def test_fit_recovers_parameters(params, tvec, noise_level):
    rng = np.random.default_rng(123)
    base = _model_from_params(params)

    q_clean = base.q(tvec)
    q_obs = q_clean if noise_level == 0.0 else q_clean * (1.0 + noise_level * rng.standard_normal(q_clean.shape))

    arps = Arps(params["di"], params["qi"], b=params["b"])
    res = arps.fit(tvec, q_obs)

    # Check closeness; hyperbolic tends to be slightly more sensitive with noise
    rtol = 5e-3 if params["mode"] != "hyperbolic" else 3e-2
    assert_allclose(res.di, params["di"], rtol=rtol, atol=rtol)
    assert_allclose(res.qi, params["qi"], rtol=rtol, atol=rtol)

    # Verify R² is high
    assert res.r2 > (0.999 if noise_level == 0.0 else 0.98)

    # Linear regression result present and sensible
    assert hasattr(res.linear, "slope")
    assert hasattr(res.linear, "intercept")

def test_linregress_transform_is_correct(params, tvec):
    base = _model_from_params(params)
    arps = Arps(params["di"], params["qi"], b=params["b"])
    q = base.q(tvec)

    lr = arps.linregress(tvec, q)
    # quick self-consistency check: build predicted transform line and correlate
    y_lin = arps.model.linearize(q)
    y_hat = lr.intercept + lr.slope * tvec
    # high correlation in transform space
    corr = np.corrcoef(y_lin, y_hat)[0, 1]
    assert corr > 0.999

# -----------------------------
# Utilities / text / simulate
# -----------------------------
def test_reader_contains_key_fields(params, tvec):
    base = _model_from_params(params)
    q = base.q(tvec)
    arps = Arps(params["di"], params["qi"], b=params["b"])
    res = arps.fit(tvec, q)

    txt = Arps.reader(res)
    assert "Decline mode is" in txt
    assert "R-squared" in txt
    assert f"exponent is {res.b}" in txt

def test_simulate_moves_params(params, tvec):
    base = _model_from_params(params)
    q = base.q(tvec) * (1.0 + 0.003 * np.random.default_rng(7).standard_normal(tvec.shape))
    arps = Arps(params["di"], params["qi"], b=params["b"])
    res = arps.fit(tvec, q)

    di_lo, qi_lo = Arps.simulate(res, prc=5.0)
    di_hi, qi_hi = Arps.simulate(res, prc=95.0)

    # monotone with prc
    assert di_lo <= di_hi or (math.isnan(di_lo) and math.isnan(di_hi))
    assert qi_lo >= qi_hi or (math.isnan(qi_lo) and math.isnan(qi_hi))