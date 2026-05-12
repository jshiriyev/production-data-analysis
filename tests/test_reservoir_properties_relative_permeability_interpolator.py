import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from prodpy.resprops.rperm._interpolator import Interpolator


@pytest.fixture
def relperm_table():
    return pd.DataFrame(
        {
            "Sw": [0.25, 0.30, 0.40, 0.50, 0.60, 0.65],
            "Krw": [0.000, 0.018, 0.092, 0.198, 0.327, 0.400],
            "Kro": [0.850, 0.754, 0.557, 0.352, 0.131, 0.000],
        }
    )


def test_pchip_interpolates_dataframe_input(relperm_table):
    interpolator = Interpolator(
        data=relperm_table,
        Sw_col="Sw",
        krw_col="Krw",
        kro_col="Kro",
    )

    krw, kro = interpolator.get(0.45)

    assert isinstance(krw, float)
    assert isinstance(kro, float)
    assert_allclose(krw, 0.14134763593380614)
    assert_allclose(kro, 0.45597239156331026)


def test_array_input_is_sorted_and_preserves_vector_shape(relperm_table):
    shuffled = relperm_table.sample(frac=1.0, random_state=1)
    interpolator = Interpolator(
        Sw_array=shuffled["Sw"],
        krw_array=shuffled["Krw"],
        kro_array=shuffled["Kro"],
    )

    krw, kro = interpolator.get(relperm_table["Sw"].to_numpy())

    assert krw.shape == (len(relperm_table),)
    assert kro.shape == (len(relperm_table),)
    assert_allclose(krw, relperm_table["Krw"])
    assert_allclose(kro, relperm_table["Kro"])


def test_spline_linear_interpolation():
    interpolator = Interpolator(
        Sw_array=[0.0, 1.0],
        krw_array=[0.0, 1.0],
        kro_array=[1.0, 0.0],
        method="spline",
        spline_order=1,
    )

    krw, kro = interpolator.get(np.array([0.25, 0.75]))

    assert_allclose(krw, [0.25, 0.75])
    assert_allclose(kro, [0.75, 0.25])


def test_extrapolated_relative_permeability_is_clipped_by_default():
    interpolator = Interpolator(
        Sw_array=[0.0, 1.0],
        krw_array=[0.0, 1.0],
        kro_array=[1.0, 0.0],
    )

    krw, kro = interpolator.get(np.array([-1.0, 2.0]))

    assert_allclose(krw, [0.0, 1.0])
    assert_allclose(kro, [1.0, 0.0])


def test_clip_can_be_disabled_for_extrapolation():
    interpolator = Interpolator(
        Sw_array=[0.0, 1.0],
        krw_array=[0.0, 1.0],
        kro_array=[1.0, 0.0],
        clip=False,
    )

    krw, kro = interpolator.get(np.array([-1.0, 2.0]))

    assert_allclose(krw, [-1.0, 2.0])
    assert_allclose(kro, [2.0, -1.0])


def test_missing_dataframe_columns_are_rejected(relperm_table):
    with pytest.raises(ValueError, match="Missing columns"):
        Interpolator(
            data=relperm_table,
            Sw_col="Sw",
            krw_col="krw",
            kro_col="Kro",
        )


def test_ambiguous_positional_and_keyword_data_are_rejected(relperm_table):
    with pytest.raises(ValueError, match="Ambiguous input"):
        Interpolator(relperm_table, data=relperm_table)


def test_invalid_positional_input_is_rejected():
    with pytest.raises(ValueError, match="Invalid positional argument"):
        Interpolator([0.0, 1.0])


def test_duplicate_saturation_values_are_rejected():
    with pytest.raises(ValueError, match="strictly increasing"):
        Interpolator(
            Sw_array=[0.0, 0.5, 0.5, 1.0],
            krw_array=[0.0, 0.2, 0.3, 1.0],
            kro_array=[1.0, 0.8, 0.7, 0.0],
        )


def test_non_one_dimensional_inputs_are_rejected_before_sorting():
    with pytest.raises(ValueError, match="Sw_array must be one-dimensional"):
        Interpolator(
            Sw_array=[[0.0, 0.5], [0.75, 1.0]],
            krw_array=[0.0, 0.2, 0.6, 1.0],
            kro_array=[1.0, 0.8, 0.4, 0.0],
        )


def test_invalid_spline_order_is_rejected():
    with pytest.raises(ValueError, match="spline_order"):
        Interpolator(
            Sw_array=[0.0, 0.5, 1.0],
            krw_array=[0.0, 0.2, 1.0],
            kro_array=[1.0, 0.8, 0.0],
            method="spline",
            spline_order=3,
        )
