import math

import pytest

from prodpy.wellflow import CircularConduit


@pytest.mark.parametrize("diameter", [-1, -10.5])
def test_init_rejects_negative_diameter(diameter):
    with pytest.raises(ValueError, match="Diameter D must be positive."):
        CircularConduit(diameter, 1)


@pytest.mark.parametrize("diameter", [0, ])
def test_init_rejects_zero_diameter(diameter):
    with pytest.raises(ValueError, match="Diameter D must satisfy 0 > D."):
        CircularConduit(diameter, 1)


@pytest.mark.parametrize("height", [-0.1, -1])
def test_init_rejects_negative_height(height):
    with pytest.raises(ValueError, match="Height h must be positive."):
        CircularConduit(10, height)


@pytest.mark.parametrize("height", [10.1])
def test_init_rejects_height_above_diameter(height):
    with pytest.raises(ValueError, match="Height h must satisfy 0 <= h <= D."):
        CircularConduit(10, height)


@pytest.mark.parametrize(
    ("diameter", "height", "message"),
    [
        (float("nan"), 1.0, "Diameter D must be finite"),
        (float("inf"), 1.0, "Diameter D must be finite"),
        (10.0, float("nan"), "Height h must be finite"),
        (10.0, float("inf"), "Height h must be finite"),
    ],
)
def test_init_rejects_non_finite_values(diameter, height, message):
    with pytest.raises(ValueError, match=message):
        CircularConduit(diameter, height)


def test_zero_height_segment_properties():
    conduit = CircularConduit(10, 0)
    segment = conduit.lower

    assert conduit.R == pytest.approx(5.0)
    assert segment.major is False
    assert segment.minor is True
    assert conduit.theta(upper=False) == pytest.approx(0.0)
    assert segment.chord == pytest.approx(0.0)
    assert segment.arc == pytest.approx(0.0)
    assert segment.perimeter == pytest.approx(0.0)
    assert segment.area == pytest.approx(0.0)

def test_semicircle_properties_match_docstring_example():
    conduit = CircularConduit(10, 5)
    segment = conduit.lower

    assert conduit.R == pytest.approx(5.0)
    assert segment.major is True
    assert segment.minor is False
    assert conduit.theta(upper=False) == pytest.approx(math.pi)
    assert segment.chord == pytest.approx(10.0)
    assert segment.arc == pytest.approx(15.707963267948966)
    assert segment.perimeter == pytest.approx(25.707963267948966)
    assert segment.area == pytest.approx(39.269908169872416)

def test_full_circle_segment_properties():
    conduit = CircularConduit(10, 10)
    segment = conduit.lower

    assert segment.major is True
    assert segment.minor is False
    assert conduit.theta(upper=False) == pytest.approx(2*math.pi)
    assert segment.chord == pytest.approx(0.0)
    assert segment.arc == pytest.approx(10 * math.pi)
    assert segment.perimeter == pytest.approx(10 * math.pi)
    assert segment.area == pytest.approx(25 * math.pi)

def test_general_case_matches_analytical_formulas():
    diameter = 10.0
    height = 3.0
    radius = diameter / 2.0
    conduit = CircularConduit(diameter, height)

    segment = conduit.lower

    expected_alpha = 2.0 * math.acos((radius - height) / radius)
    expected_chord = 2.0 * math.sqrt(2.0 * radius * height - height**2)
    expected_arc = expected_alpha * radius
    expected_area = radius**2 / 2 * expected_alpha - (radius - height) * expected_chord / 2.0

    assert segment.major is False
    assert segment.minor is True
    assert conduit.theta(upper=False) == pytest.approx(expected_alpha)
    assert segment.chord == pytest.approx(expected_chord)
    assert segment.arc == pytest.approx(expected_arc)
    assert segment.perimeter == pytest.approx(segment.arc + segment.chord)
    assert segment.area == pytest.approx(expected_area)


def test_upper_and_lower_segments_partition_circle():
    conduit = CircularConduit(10, 3)

    assert conduit.upper.chord == pytest.approx(conduit.lower.chord)
    assert conduit.upper.arc + conduit.lower.arc == pytest.approx(conduit.P)
    assert conduit.upper.area + conduit.lower.area == pytest.approx(conduit.A)
    assert conduit.upper.major is True
    assert conduit.lower.major is False


def test_call_with_zero_added_area_keeps_height_and_area():
    conduit = CircularConduit(10, 5)
    initial_height = conduit.h
    initial_area = conduit.lower.area

    conduit(0.0)

    assert conduit.h == pytest.approx(initial_height)
    assert conduit.lower.area == pytest.approx(initial_area)


def test_call_with_positive_added_area_updates_height_and_area():
    conduit = CircularConduit(10, 5)
    initial_height = conduit.h
    initial_area = conduit.lower.area

    conduit(1.0)

    assert conduit.h > initial_height
    assert conduit.lower.area == pytest.approx(initial_area + 1.0)


@pytest.mark.parametrize(
    ("added_area", "message"),
    [
        (-1.0, "Added area must be positive."),
        (float("nan"), "Added area must be finite"),
        (float("inf"), "Added area must be finite"),
    ],
)
def test_call_rejects_invalid_added_area(added_area, message):
    conduit = CircularConduit(10, 5)

    with pytest.raises(ValueError, match=message):
        conduit(added_area)


def test_call_rejects_when_conduit_would_fill():
    conduit = CircularConduit(10, 5)
    remaining_area = conduit.A - conduit.lower.area

    with pytest.raises(ValueError, match="The cross-section is filled."):
        conduit(remaining_area)
