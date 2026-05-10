import numpy as np
import pytest

from prodpy.porsim.prerun._constraint import Constraint
from prodpy.porsim.prerun._boundary import Boundary
from prodpy.porsim.prerun._borehole import Borehole


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


def test_borehole_initializes_well_properties():
    well = Borehole(index=(0, 1), axis="z", radius=0.25, skin=2.0)

    assert well.index == (0, 1)
    assert well.axis == "z"
    assert well.radius == pytest.approx(0.25)
    assert well.skin == pytest.approx(2.0)
    assert well.constraints == ()


def test_borehole_adds_constraints_from_objects_and_kwargs_sorted_by_start():
    well = Borehole(
        index=(0, 1),
        constraints=[
            Constraint(start=10.0, stop=20.0, press=3000.0),
            Constraint(start=0.0, stop=10.0, orate=1000.0),
        ],
    )

    added = well.add_constraint(start=20.0, wrate=500.0)

    assert [constraint.start for constraint in well.constraints] == [0.0, 10.0, 20.0]
    assert well.constraints[-1] is added
    assert well.active_constraint(5.0).mode == "orate"
    assert well.active_constraint(10.0).mode == "press"
    assert well.active_constraint(20.0).mode == "wrate"


def test_borehole_rejects_overlapping_constraints():
    well = Borehole(constraints=[Constraint(start=0.0, stop=10.0, orate=1000.0)])

    with pytest.raises(ValueError, match="overlaps with an existing constraint"):
        well.add_constraint(start=9.0, stop=20.0, press=3000.0)


def test_borehole_allows_touching_half_open_constraints():
    well = Borehole()

    well.add_constraint(start=0.0, stop=10.0, orate=1000.0)
    well.add_constraint(start=10.0, stop=20.0, press=3000.0)

    assert well.validate_schedule()
    assert well.active_constraint(9.999).mode == "orate"
    assert well.active_constraint(10.0).mode == "press"


def test_borehole_rejects_overlapping_open_ended_constraints():
    well = Borehole(constraints=[Constraint(start=5.0, orate=1000.0)])

    with pytest.raises(ValueError, match="overlaps with an existing constraint"):
        well.add_constraint(start=10.0, press=3000.0)


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"index": []}, TypeError, "index must be a tuple or None"),
        ({"index": ()}, ValueError, "index cannot be an empty tuple"),
        ({"index": (1, "2")}, TypeError, "all index values must be integers"),
        ({"axis": "u"}, ValueError, "axis must be one of"),
        ({"axis": 1}, TypeError, "axis must be a string"),
        ({"radius": 0.0}, ValueError, "radius must be positive"),
        ({"radius": "bad"}, ValueError, "radius must be convertible to float"),
        ({"skin": "bad"}, ValueError, "skin must be convertible to float"),
    ],
)
def test_borehole_invalid_properties_raise(kwargs, error_type, message):
    with pytest.raises(error_type, match=message):
        Borehole(**kwargs)


def test_borehole_constraints_are_read_only_tuple_snapshot():
    well = Borehole()
    constraint = well.add_constraint(press=4000.0)

    constraints = well.constraints

    assert constraints == (constraint,)
    with pytest.raises(AttributeError):
        constraints.append(Constraint(start=1.0, orate=100.0))


def test_borehole_schedule_and_change_times():
    well = Borehole(
        constraints=[
            Constraint(start=0.0, stop=10.0, orate=1000.0),
            Constraint(start=10.0, press=3000.0),
        ],
    )

    schedule = well.schedule()
    rows = schedule.to_dict("records") if hasattr(schedule, "to_dict") else schedule

    assert rows[0] == {
        "index": 0,
        "start_days": 0.0,
        "stop_days": 10.0,
        "mode": "orate",
        "limit": 1000.0,
        "unit": "bbl/day",
    }
    assert rows[1]["index"] == 1
    assert rows[1]["start_days"] == 10.0
    assert rows[1]["stop_days"] is None or np.isnan(rows[1]["stop_days"])
    assert rows[1]["mode"] == "press"
    assert rows[1]["limit"] == 3000.0
    assert rows[1]["unit"] == "psi"
    assert well.change_times() == [0.0, 10.0]


def test_borehole_repr_uses_index_not_missing_well_attribute():
    well = Borehole(index=(0, 1), constraints=[Constraint(press=4000.0)])

    assert repr(well) == "Borehole(index=(0, 1), radius=0.5 ft, skin=0, constraints=1)"


def test_edge_bound_initializes_constraint_and_face_properties():
    edge = Boundary("xmin", press=500.0)

    assert edge.face == "xmin"
    assert edge.axis == "x"
    assert edge.mode == "press"
    assert edge.limit == pytest.approx(500.0)
