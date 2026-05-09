import numpy as np
import pytest

from prodpy.porsim.prerun._constraint import Constraint
from prodpy.porsim.prerun._edge_bound import EdgeBound
from prodpy.porsim.prerun._well_bound import WellBound


@pytest.mark.parametrize(
    ("kwargs", "mode", "limit"),
    [
        ({"press": 4000.0}, "press", 4000.0),
        ({"lrate": 750.0}, "lrate", 750.0),
        ({"orate": 500.0}, "orate", 500.0),
        ({"wrate": -300.0}, "wrate", -300.0),
        ({"grate": 100_000.0}, "grate", 100_000.0),
    ],
)
def test_constraint_limit_roundtrip(kwargs, mode, limit):
    constraint = Constraint(**kwargs)

    assert constraint.mode == mode
    assert constraint.limit == pytest.approx(limit)


def test_constraint_time_window_defaults_to_active_from_zero():
    constraint = Constraint(press=4000.0)

    assert constraint.start == pytest.approx(0.0)
    assert constraint.stop is None
    assert constraint.is_active(0.0)
    assert constraint.is_active(100.0)


def test_constraint_is_active_uses_inclusive_start_exclusive_stop():
    constraint = Constraint(press=4000.0, start=2.0, stop=5.0)

    assert not constraint.is_active(1.999)
    assert constraint.is_active(2.0)
    assert constraint.is_active(4.999)
    assert not constraint.is_active(5.0)


def test_constraint_start_cannot_be_changed_to_overlap_stop():
    constraint = Constraint(press=4000.0, start=1.0, stop=3.0)

    with pytest.raises(ValueError, match="start must be less than stop"):
        constraint.start = 3.0


def test_constraint_stop_can_be_cleared():
    constraint = Constraint(press=4000.0, start=1.0, stop=3.0)

    constraint.stop = None

    assert constraint.stop is None
    assert constraint.is_active(100.0)


def test_constraint_set_limit_replaces_mode_and_limit():
    constraint = Constraint(press=4000.0)

    constraint.set_limit(orate=500.0)

    assert constraint.mode == "orate"
    assert constraint.limit == pytest.approx(500.0)


def test_constraint_productivity_roundtrip_scalar_ft3_per_day_per_psi():
    constraint = Constraint(press=4000.0)

    constraint.prod = 10.0

    assert constraint.prod == pytest.approx(10.0)


def test_constraint_productivity_roundtrip_array_ft3_per_day_per_psi():
    constraint = Constraint(press=4000.0)
    productivity = np.array([5.0, 10.0, 15.0])

    constraint.prod = productivity

    np.testing.assert_allclose(constraint.prod, productivity)


def test_constraint_unassigned_productivity_raises():
    constraint = Constraint(press=4000.0)

    with pytest.raises(ValueError, match="productivity has not been assigned yet"):
        _ = constraint.prod


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "No condition is provided"),
        ({"press": 4000.0, "orate": 500.0}, "Multiple conditions are provided"),
        ({"foo": 1.0}, "Invalid constraint type"),
        ({"press": "bad"}, "Limit value must be convertible to float"),
    ],
)
def test_constraint_invalid_limits_raise(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Constraint(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"press": 4000.0, "start": -1.0}, "start must be non-negative"),
        ({"press": 4000.0, "stop": -1.0}, "stop must be non-negative"),
        ({"press": 4000.0, "start": 1.0, "stop": 1.0}, "stop must be greater than start"),
        ({"press": 4000.0, "start": 2.0, "stop": 1.0}, "stop must be greater than start"),
        ({"press": 4000.0, "start": "bad"}, "start must be convertible to float"),
        ({"press": 4000.0, "stop": "bad"}, "stop must be convertible to float"),
    ],
)
def test_constraint_invalid_time_window_raises(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Constraint(**kwargs)


def test_well_bound_initializes_constraint_and_well_properties():
    well = WellBound((0, 1), axis="z", radius=0.25, skin=2.0, press=4000.0)

    assert well.index == (0, 1)
    assert well.axis == "z"
    assert well.radius == pytest.approx(0.25)
    assert well.skin == pytest.approx(2.0)
    assert well.mode == "press"
    assert well.limit == pytest.approx(4000.0)


def test_edge_bound_initializes_constraint_and_face_properties():
    edge = EdgeBound("xmin", press=500.0)

    assert edge.face == "xmin"
    assert edge.axis == "x"
    assert edge.mode == "press"
    assert edge.limit == pytest.approx(500.0)
