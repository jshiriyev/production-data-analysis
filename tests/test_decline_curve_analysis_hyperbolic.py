# tests/test_hyperbolic.py
import math
import numpy as np
import pytest
from numpy.testing import assert_allclose

# ---- adjust this import to your module name/path ----
from prodpy.decline import Hyperbolic


@pytest.fixture
def params():
    # Use "safe" b values (not exactly 0 or 1) because the current implementation
    # does not special-case those limits.
    return dict(b=0.5, di=0.25, qi=120.0)

@pytest.fixture
def model(params):
    return Hyperbolic(**params)

@pytest.fixture
def tvec():
    # Positive times in the same unit as di
    return np.linspace(0.0, 10.0, 201)

# -----------------------------
# Basic formula / shape checks
# -----------------------------
def test_q_at_t0_equals_qi(model):
    q0 = model.q(0.0)
    # q() returns ndarray for array-like inputs; for scalar, NumPy returns a 0-d array.
    # Convert to float for comparison.
    q0 = float(np.asarray(q0))
    assert math.isclose(q0, model.qi, rel_tol=0, abs_tol=1e-12)

def test_q_monotone_decline(model, tvec):
    q = model.q(tvec)
    # strictly non-increasing (allowing tiny numeric jitter)
    assert np.all(np.diff(q) <= 1e-12)

def test_d_at_t0_equals_di(model):
    d0 = float(np.asarray(model.d(0.0)))
    assert math.isclose(d0, model.di, rel_tol=0, abs_tol=1e-12)

def test_d_monotone_decreasing(model, tvec):
    d = model.d(tvec)
    assert np.all(np.diff(d) <= 1e-12)

def test_N_zero_at_t0_and_increasing(model, tvec):
    N = model.N(tvec)
    assert math.isclose(float(np.asarray(model.N(0.0))), 0.0, rel_tol=0, abs_tol=1e-12)
    assert np.all(np.diff(N) >= -1e-12)  # non-decreasing

def test_vectorization_shapes(model, tvec):
    q = model.q(tvec)
    d = model.d(tvec)
    D = model.D(tvec)
    N = model.N(tvec)
    assert q.shape == tvec.shape
    assert d.shape == tvec.shape
    assert D.shape == tvec.shape
    assert N.shape == tvec.shape

# ------------------------------------
# Calculus / identity consistency
# ------------------------------------
def test_d_is_minus_qprime_over_q(model, tvec):
    # Numerical derivative of q(t) via central differences
    q = model.q(tvec)
    dt = tvec[1] - tvec[0]
    dqdt = np.gradient(q, dt, edge_order=2)
    left = -dqdt / q
    right = model.d(tvec)
    # allow a small tolerance for numerical gradient errors
    assert_allclose(left, right, rtol=2e-4, atol=2e-4)

def test_dNdt_equals_q(model, tvec):
    N = model.N(tvec)
    dt = tvec[1] - tvec[0]
    dNdt = np.gradient(N, dt, edge_order=2)
    q = model.q(tvec)
    assert_allclose(dNdt, q, rtol=2e-4, atol=2e-4)

# ------------------------------------
# Economic limit consistency
# ------------------------------------
@pytest.mark.parametrize("qec", [40.0, 20.0, 10.0])
def test_T_satisfies_q_of_T_equals_qec(model, qec):
    T = model.T(qec)
    q_at_T = float(np.asarray(model.q(T)))
    assert_allclose(q_at_T, qec, rtol=1e-9, atol=1e-9)

@pytest.mark.parametrize("qec", [40.0, 20.0, 10.0])
def test_Nec_equals_N_at_T(model, qec):
    T = model.T(qec)
    N_at_T = float(np.asarray(model.N(T)))
    N_ec = model.N_ec(qec)
    assert_allclose(N_ec, N_at_T, rtol=1e-10, atol=1e-10)

# ------------------------------------
# Linearization & regression
# ------------------------------------
def test_linearization_is_linear_in_t(params):
    b, di, qi = params["b"], params["di"], params["qi"]
    m = Hyperbolic(b=b, di=di, qi=qi)

    t = np.linspace(0.0, 12.0, 100)
    q = m.q(t)
    y = m.linearize(q)  # y = q^{-b}

    # Theoretical: y = c + m_t * t with c = qi^{-b}, m_t = (b*di) * qi^{-b}
    c_th = qi ** (-b)
    mt_th = (b * di) * (qi ** (-b))

    # Fit a line y ≈ a + s t
    A = np.vstack([np.ones_like(t), t]).T
    a_fit, s_fit = np.linalg.lstsq(A, y, rcond=None)[0]

    assert_allclose(a_fit, c_th, rtol=1e-10, atol=1e-10)
    assert_allclose(s_fit, mt_th, rtol=1e-10, atol=1e-10)

def test_fit_recovers_parameters(params):
    rng = np.random.default_rng(7)
    b, di, qi = params["b"], params["di"], params["qi"]
    m_true = Hyperbolic(b=b, di=di, qi=qi)

    t = np.linspace(0.0, 10.0, 80)
    q_clean = m_true.q(t)

    # Add small relative noise to mimic measurement noise
    noise = 0.002 * rng.standard_normal(q_clean.shape)  # 0.2% noise
    q_noisy = q_clean * (1.0 + noise)

    m_est = Hyperbolic(b=b, di=1.0, qi=1.0).fit(t, q_noisy)

    # di and qi should be close to the true values
    assert_allclose(m_est.di, di, rtol=5e-2, atol=5e-2)
    assert_allclose(m_est.qi, qi, rtol=5e-2, atol=5e-2)

# ------------------------------------
# API behavior & representation
# ------------------------------------
def test_with_params_returns_new_instance(model):
    m2 = model.with_params(di=model.di * 2, qi=model.qi * 0.5)
    assert m2 is not model
    assert math.isclose(m2.di, model.di * 2, rel_tol=0, abs_tol=0)
    assert math.isclose(m2.qi, model.qi * 0.5, rel_tol=0, abs_tol=0)

def test_mode_property(model):
    assert model.mode == "hyperbolic"

def test_repr_contains_fields(model):
    s = repr(model)
    assert "Hyperbolic(" in s and "b=" in s and "di=" in s and "qi=" in s

# ------------------------------------
# Error handling
# ------------------------------------
def test_invalid_b_raises():
    with pytest.raises(ValueError):
        Hyperbolic(b=-1.0, di=0.2, qi=10.0)
    with pytest.raises(ValueError):
        Hyperbolic(b=1.1, di=0.2, qi=10.0)

def test_nonpositive_di_or_qi_raises():
    with pytest.raises(ValueError):
        Hyperbolic(b=0.3, di=0.0, qi=10.0)
    with pytest.raises(ValueError):
        Hyperbolic(b=0.3, di=-0.2, qi=10.0)
    with pytest.raises(ValueError):
        Hyperbolic(b=0.3, di=0.2, qi=0.0)
    with pytest.raises(ValueError):
        Hyperbolic(b=0.3, di=0.2, qi=-5.0)

def test_nonpositive_qec_raises(model):
    with pytest.raises(ValueError):
        model.T(0.0)
    with pytest.raises(ValueError):
        model.N_ec(-1.0)

def test_fit_shape_mismatch_raises(model):
    t = np.linspace(0, 5, 10)
    q = np.linspace(100, 90, 9)
    with pytest.raises(ValueError):
        model.fit(t, q)
