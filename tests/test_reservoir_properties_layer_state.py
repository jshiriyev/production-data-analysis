import importlib.util
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose


def _load_layer_module():
    module_path = Path(__file__).parents[1] / "prodpy" / "resprops" / "_layer.py"
    spec = importlib.util.spec_from_file_location("layer", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


Layer = _load_layer_module().Layer


def test_reduced_permeability_and_si_storage():
    layer = Layer(
        [10.0, 15.0, 20.0],
        poro=[0.1, 0.2, 0.3],
        comp=1e-6,
        press=3000.0,
        yreduce=0.5,
        zreduce=0.1,
    )

    assert_allclose(layer.xperm, [10.0, 15.0, 20.0])
    assert_allclose(layer.yperm, [5.0, 7.5, 10.0])
    assert_allclose(layer.zperm, [1.0, 1.5, 2.0])
    assert_allclose(
        layer.perm,
        [
            [10.0, 5.0, 1.0],
            [15.0, 7.5, 1.5],
            [20.0, 10.0, 2.0],
        ],
    )

    assert_allclose(layer._xperm, np.array([10.0, 15.0, 20.0]) * layer.MD_TO_M2)
    assert_allclose(layer._perm, layer.perm * layer.MD_TO_M2)
    assert_allclose(layer.poro, [0.1, 0.2, 0.3])
    assert_allclose(layer.comp, [1e-6])
    assert_allclose(layer._comp, np.array([1e-6]) / layer.PSI_TO_PA)
    assert_allclose(layer.press, [3000.0])
    assert_allclose(layer._press, np.array([3000.0]) * layer.PSI_TO_PA)


def test_explicit_permeability_overrides_reduction_and_broadcasts_scalars():
    layer = Layer(
        [10.0, 20.0, 30.0],
        yperm=5.0,
        zperm=[3.0, 2.0, 1.0],
        yreduce=0.25,
        zreduce=0.25,
    )

    assert_allclose(layer.yperm, [5.0, 5.0, 5.0])
    assert_allclose(layer.zperm, [3.0, 2.0, 1.0])
    assert_allclose(
        layer.perm,
        [
            [10.0, 5.0, 3.0],
            [20.0, 5.0, 2.0],
            [30.0, 5.0, 1.0],
        ],
    )


def test_set_permeability_replaces_directional_values():
    layer = Layer(1.0)

    layer.set_permeability([10.0, 20.0], yperm=[5.0, 6.0], zperm=2.0)

    assert_allclose(layer.perm, [[10.0, 5.0, 2.0], [20.0, 6.0, 2.0]])


def test_optional_properties_default_to_none():
    layer = Layer(10.0)

    assert layer.poro is None
    assert layer.comp is None
    assert layer.press is None


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: Layer(-1.0), "xperm cannot be smaller than 0"),
        (lambda: Layer(1.0, yperm=-1.0), "yperm cannot be smaller than 0"),
        (lambda: Layer(1.0, zperm=-1.0), "zperm cannot be smaller than 0"),
        (lambda: Layer(1.0, yreduce=-0.1), "yreduce cannot be smaller than 0"),
        (lambda: Layer(1.0, zreduce=-0.1), "zreduce cannot be smaller than 0"),
        (lambda: Layer(1.0, poro=-0.1), "poro cannot be smaller than 0"),
        (lambda: Layer(1.0, poro=1.1), "poro cannot be larger than 1"),
        (lambda: Layer(1.0, comp=-1e-6), "comp cannot be smaller than 0"),
        (lambda: Layer(1.0, press=-100.0), "press cannot be smaller than 0"),
    ],
)
def test_invalid_physical_ranges_are_rejected(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Layer([]),
        lambda: Layer(1.0, yperm=[]),
        lambda: Layer(1.0, zperm=[]),
        lambda: Layer(1.0, poro=[]),
        lambda: Layer(1.0, comp=[]),
        lambda: Layer(1.0, press=[]),
    ],
)
def test_empty_values_are_rejected(factory):
    with pytest.raises(ValueError, match="cannot be empty"):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Layer(float("inf")),
        lambda: Layer(float("-inf")),
        lambda: Layer(1.0, yperm=float("inf")),
        lambda: Layer(1.0, zperm=float("-inf")),
        lambda: Layer(1.0, comp=float("inf")),
        lambda: Layer(1.0, press=float("-inf")),
    ],
)
def test_infinite_values_are_rejected(factory):
    with pytest.raises(ValueError, match="must not contain positive or negative infinity"):
        factory()


def test_nan_values_are_accepted():
    layer = Layer(
        [10.0, np.nan],
        yperm=np.nan,
        zperm=[1.0, np.nan],
        poro=[0.2, np.nan],
        comp=np.nan,
        press=np.nan,
    )

    assert_allclose(layer.xperm, [10.0, np.nan], equal_nan=True)
    assert_allclose(layer.yperm, [np.nan, np.nan], equal_nan=True)
    assert_allclose(layer.zperm, [1.0, np.nan], equal_nan=True)
    assert_allclose(
        layer.perm,
        [
            [10.0, np.nan, 1.0],
            [np.nan, np.nan, np.nan],
        ],
        equal_nan=True,
    )
    assert_allclose(layer.poro, [0.2, np.nan], equal_nan=True)
    assert_allclose(layer.comp, [np.nan], equal_nan=True)
    assert_allclose(layer.press, [np.nan], equal_nan=True)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"yperm": [1.0, 2.0]}, "yperm must be scalar or have length 3; got length 2."),
        ({"zperm": [1.0, 2.0]}, "zperm must be scalar or have length 3; got length 2."),
    ],
)
def test_directional_permeability_lengths_must_match_xperm(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Layer([10.0, 20.0, 30.0], **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"yreduce": [0.5, 0.6]},
        {"zreduce": [0.1, 0.2]},
    ],
)
def test_reduction_ratios_must_be_scalar(kwargs):
    with pytest.raises(ValueError):
        Layer([10.0, 20.0], **kwargs)
