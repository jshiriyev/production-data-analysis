# tests/test_exponential.py
import math
import numpy as np
import pytest
from numpy.testing import assert_allclose

# ---- adjust this import to your module name/path ----
from prodpy.decline import Exponential

# Optional: cross-check vs Hyperbolic with small b
from prodpy.decline import Hyperbolic  # pragma: no cover

@pytest.fixture
def params():
    return dict(di=0.25, qi=120.0)


@pytest.fixture
def model(params):
    return Exponential(**params)


@pytest.fixture
def tvec():
    return np.linspace(0.0, 10.0, 4001)


# -----------------------------
# Basic formula / shape checks
# -----------------------------
def test_q_at_t0_equals_qi(model):
    q0 = float(np.asarray(model.q(0.0)))
    assert math.isclose(q0, model.qi, rel_tol=0, abs_tol=1e-12)


def test_q_monotone_decline(model, tvec):
    q = model.q(tvec)
    assert np.all(np.diff(q) <= 1e-12)


def test_d_is_constant_equals_di(model, tvec):
    d = model.d(tvec)
    assert_allclose(d, np.full_like(tvec, model.di, dtype=float), rtol=0, atol=0)


def test_D_is_constant(model, tvec):
    D = model.D(tvec)
    expected = 1.0 - math.exp(-model.di)
    assert_allclose(D, np.full_like(tvec, expected, dtype=float), rtol=0, atol=0)


def test_N_zero_at_t0_and_increasing(model, tvec):
    N = model.N(tvec)
    assert math.isclose(float(np.asarray(model.N(0.0))), 0.0, rel_tol=0, abs_tol=1e-12)
    assert np.all(np.diff(N) >= -1e-12)


def test_vectorization_shapes(model, tvec):
    assert model.q(tvec).shape == tvec.shape
    assert model.d(tvec).shape == tvec.shape
    assert model.D(tvec).shape == tvec.shape
    assert model.N(tvec).shape == tvec.shape


# ------------------------------------
# Calculus / identity consistency
# ------------------------------------
def test_d_is_minus_qprime_over_q(model, tvec):
    q = model.q(tvec)
    dt = tvec[1] - tvec[0]
    dqdt = np.gradient(q, dt, edge_order=2)
    left = -dqdt / q
    right = model.d(tvec)
    assert_allclose(left, right, rtol=3e-5, atol=3e-5)


def test_dNdt_equals_q(model, tvec):
    N = model.N(tvec)
    dt = tvec[1] - tvec[0]
    dNdt = np.gradient(N, dt, edge_order=2)
    q = model.q(tvec)
    assert_allclose(dNdt, q, rtol=2e-6, atol=2e-6)


# ------------------------------------
# Economic limit consistency
# ------------------------------------
@pytest.mark.parametrize("qec", [40.0, 20.0, 10.0])
def test_T_satisfies_q_of_T_equals_qec(model, qec):
    T = model.T(qec)
    q_at_T = float(np.asarray(model.q(T)))
    assert_allclose(q_at_T, qec, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("qec", [40.0, 20.0, 10.0])
def test_Nec_equals_N_at_T(model, qec):
    T = model.T(qec)
    N_at_T = float(np.asarray(model.N(T)))
    N_ec = model.N_ec(qec)
    assert_allclose(N_ec, N_at_T, rtol=1e-12, atol=1e-12)


# ------------------------------------
# Linearization & regression
# ------------------------------------
def test_linearization_is_linear_in_t(params):
    di, qi = params["di"], params["qi"]
    m = Exponential(di=di, qi=qi)

    t = np.linspace(0.0, 12.0, 100)
    q = m.q(t)
    y = m.linearize(q)  # y = ln(q)

    # Theoretical: y = ln(qi) - di * t
    c_th = math.log(qi)
    m_th = -di

    A = np.vstack([np.ones_like(t), t]).T
    a_fit, s_fit = np.linalg.lstsq(A, y, rcond=None)[0]

    assert_allclose(a_fit, c_th, rtol=1e-12, atol=1e-12)
    assert_allclose(s_fit, m_th, rtol=1e-12, atol=1e-12)


def test_fit_recovers_parameters(params):
    rng = np.random.default_rng(7)
    di, qi = params["di"], params["qi"]
    m_true = Exponential(di=di, qi=qi)

    t = np.linspace(0.0, 10.0, 120)
    q_clean = m_true.q(t)

    # Multiplicative noise (so ln(q) noise is roughly additive/small)
    noise = 0.002 * rng.standard_normal(q_clean.shape)
    q_noisy = q_clean * (1.0 + noise)

    m_est = Exponential(di=1.0, qi=1.0).fit(t, q_noisy)

    assert_allclose(m_est.di, di, rtol=5e-3, atol=5e-3)
    assert_allclose(m_est.qi, qi, rtol=5e-3, atol=5e-3)


# ------------------------------------
# Cross-check vs Hyperbolic (small b)
# ------------------------------------
def test_matches_hyperbolic_small_b(params, tvec):
    di, qi = params["di"], params["qi"]
    m_exp = Exponential(di=di, qi=qi)
    m_hyp = Hyperbolic(b=1e-6, di=di, qi=qi)  # very small b

    q_exp = m_exp.q(tvec)
    q_hyp = m_hyp.q(tvec)

    N_exp = m_exp.N(tvec)
    N_hyp = m_hyp.N(tvec)

    # Loose tolerance to accommodate finite-precision power at very small b
    assert_allclose(q_hyp, q_exp, rtol=2e-5, atol=2e-7)
    assert_allclose(N_hyp, N_exp, rtol=2e-5, atol=2e-7)


# ------------------------------------
# API behavior & representation
# ------------------------------------
def test_with_params_returns_new_instance(model):
    m2 = model.with_params(di=model.di * 2, qi=model.qi * 0.5)
    assert m2 is not model
    assert math.isclose(m2.di, model.di * 2, rel_tol=0, abs_tol=0)
    assert math.isclose(m2.qi, model.qi * 0.5, rel_tol=0, abs_tol=0)


def test_mode_property(model):
    assert model.mode == "exponential"


def test_repr_contains_fields(model):
    s = repr(model)
    assert "Exponential(" in s and "di=" in s and "qi=" in s


# ------------------------------------
# Error handling
# ------------------------------------
def test_nonpositive_di_or_qi_raises():
    with pytest.raises(ValueError):
        Exponential(di=0.0, qi=10.0)
    with pytest.raises(ValueError):
        Exponential(di=-0.2, qi=10.0)
    with pytest.raises(ValueError):
        Exponential(di=0.2, qi=0.0)
    with pytest.raises(ValueError):
        Exponential(di=0.2, qi=-5.0)


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
