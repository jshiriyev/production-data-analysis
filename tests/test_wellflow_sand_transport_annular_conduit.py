import math

import pytest

from prodpy.wellflow import AnnularConduit, CircularConduit


def corrected_inner_height(outer_diameter: float, inner_diameter: float, height: float) -> float:
    return min(max(height - (outer_diameter - inner_diameter) / 2.0, 0.0), inner_diameter)


@pytest.mark.parametrize("inner_diameter", [10.0, 12.0])
def test_init_rejects_inner_diameter_not_less_than_outer(inner_diameter):
    with pytest.raises(ValueError, match="Inner Diameter Di must satisfy 0 < Di < Do."):
        AnnularConduit(10.0, inner_diameter, 5.0)


@pytest.mark.parametrize(
    ("inner_diameter", "message"),
    [
        (-1.0, "Diameter D must be positive."),
        (float("nan"), "Diameter D must be finite"),
        (float("inf"), "Diameter D must be finite"),
    ],
)
def test_init_rejects_invalid_inner_diameter(inner_diameter, message):
    with pytest.raises(ValueError, match=message):
        AnnularConduit(10.0, inner_diameter, 5.0)


@pytest.mark.parametrize(
    ("height", "expected"),
    [
        (0.0, 0.0),
        (2.0, 0.0),
        (3.0, 0.0),
        (5.0, 2.0),
        (7.0, 4.0),
        (10.0, 4.0),
    ],
)
def test_height_correction_clamps_to_inner_pipe_bounds(height, expected):
    annular = AnnularConduit(10.0, 4.0, 5.0)

    assert annular.height_correction(height) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("height", "expected_lower_area", "expected_upper_area", "expected_lower_arc", "expected_upper_arc"),
    [
        (0.0, 0.0, math.pi * (10.0**2 - 4.0**2) / 4.0, 0.0, math.pi * (10.0 + 4.0)),
        (10.0, math.pi * (10.0**2 - 4.0**2) / 4.0, 0.0, math.pi * (10.0 + 4.0), 0.0),
    ],
)
def test_annular_boundary_cases_match_full_and_empty_segments(
    height,
    expected_lower_area,
    expected_upper_area,
    expected_lower_arc,
    expected_upper_arc,
):
    annular = AnnularConduit(10.0, 4.0, height)

    assert annular.chord() == pytest.approx(0.0)
    assert annular.lower.area == pytest.approx(expected_lower_area)
    assert annular.upper.area == pytest.approx(expected_upper_area)
    assert annular.lower.arc == pytest.approx(expected_lower_arc)
    assert annular.upper.arc == pytest.approx(expected_upper_arc)


def test_annular_mid_height_is_symmetric():
    annular = AnnularConduit(10.0, 4.0, 5.0)

    assert annular.A == pytest.approx(math.pi * (10.0**2 - 4.0**2) / 4.0)
    assert annular.P == pytest.approx(math.pi * (10.0 + 4.0))
    assert annular.chord() == pytest.approx(6.0)
    assert annular.lower.area == pytest.approx(annular.A / 2.0)
    assert annular.upper.area == pytest.approx(annular.A / 2.0)
    assert annular.lower.arc == pytest.approx(annular.P / 2.0)
    assert annular.upper.arc == pytest.approx(annular.P / 2.0)


@pytest.mark.parametrize("height", [2.0, 4.0, 7.0])
def test_annular_geometry_matches_outer_minus_inner_circles(height):
    outer_diameter = 10.0
    inner_diameter = 4.0
    annular = AnnularConduit(outer_diameter, inner_diameter, height)

    outer = CircularConduit(outer_diameter, height)
    inner_height = corrected_inner_height(outer_diameter, inner_diameter, height)
    inner = CircularConduit(inner_diameter, inner_height)

    expected_chord = outer.chord() - inner.chord()
    expected_lower_arc = outer.lower.arc + inner.lower.arc
    expected_upper_arc = outer.upper.arc + inner.upper.arc
    expected_lower_area = outer.lower.area - inner.lower.area
    expected_upper_area = outer.upper.area - inner.upper.area

    assert annular.chord() == pytest.approx(expected_chord)
    assert annular.lower.chord == pytest.approx(expected_chord)
    assert annular.upper.chord == pytest.approx(expected_chord)
    assert annular.lower.arc == pytest.approx(expected_lower_arc)
    assert annular.upper.arc == pytest.approx(expected_upper_arc)
    assert annular.lower.area == pytest.approx(expected_lower_area)
    assert annular.upper.area == pytest.approx(expected_upper_area)


def test_upper_and_lower_segments_partition_annulus():
    annular = AnnularConduit(10.0, 4.0, 3.0)

    assert annular.upper.chord == pytest.approx(annular.lower.chord)
    assert annular.upper.arc + annular.lower.arc == pytest.approx(annular.P)
    assert annular.upper.area + annular.lower.area == pytest.approx(annular.A)
    assert annular.upper.major is True
    assert annular.lower.major is False


def test_call_with_positive_added_area_updates_height_and_area():
    annular = AnnularConduit(10.0, 4.0, 5.0)
    initial_height = annular.h
    initial_area = annular.lower.area

    annular(1.0)

    assert annular.h > initial_height
    assert annular.lower.area == pytest.approx(initial_area + 1.0)


@pytest.mark.parametrize(
    ("added_area", "message"),
    [
        (-1.0, "Added area must be positive."),
        (float("nan"), "Added area must be finite"),
        (float("inf"), "Added area must be finite"),
    ],
)
def test_call_rejects_invalid_added_area(added_area, message):
    annular = AnnularConduit(10.0, 4.0, 5.0)

    with pytest.raises(ValueError, match=message):
        annular(added_area)


def test_call_rejects_when_annulus_would_fill():
    annular = AnnularConduit(10.0, 4.0, 5.0)
    remaining_area = annular.A - annular.lower.area

    with pytest.raises(ValueError, match="The cross-section is filled."):
        annular(remaining_area)
