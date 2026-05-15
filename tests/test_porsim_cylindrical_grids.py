from functools import cached_property

import numpy as np
import pytest

from prodpy.porsim.cylindrical._base_class import BaseClass
from prodpy.porsim.cylindrical._radial_grids import RadialGrids


def test_cylindrical_base_converts_public_lengths_to_internal_si_values():
    grid = BaseClass(
        radii=[10.0, 20.0],
        zdelta=[5.0, 15.0],
        tdelta=[np.pi, np.pi],
        depth=1000.0,
    )

    np.testing.assert_allclose(grid.radii, [10.0, 20.0])
    np.testing.assert_allclose(
        grid._radii,
        np.array([10.0, 20.0]) * BaseClass.FEET_TO_METERS,
    )
    np.testing.assert_allclose(grid.zdelta, [5.0, 15.0])
    np.testing.assert_allclose(
        grid._zdelta,
        np.array([5.0, 15.0]) * BaseClass.FEET_TO_METERS,
    )
    np.testing.assert_allclose(grid.tdelta, [np.pi, np.pi])
    np.testing.assert_allclose(grid.depth, 1000.0)
    np.testing.assert_allclose(grid._depth, np.asarray(1000.0) * BaseClass.FEET_TO_METERS)
    assert grid.depth.shape == ()


def test_cylindrical_base_uses_full_circle_default_theta_spacing():
    grid = BaseClass(radii=[10.0, 20.0], zdelta=5.0)

    np.testing.assert_allclose(grid.tdelta, [2 * np.pi])


@pytest.mark.parametrize("field", ["radii", "zdelta", "tdelta"])
@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([], "cannot be empty"),
        ([1.0, np.nan], "finite"),
        ([1.0, np.inf], "finite"),
        ([1.0, 0.0], "positive"),
        ([1.0, -1.0], "positive"),
        (["bad"], "convertible"),
        ([[1.0, 2.0]], "scalar or one-dimensional"),
    ],
)
def test_cylindrical_base_rejects_invalid_axis_like_inputs(field, value, message):
    kwargs = {
        "radii": [10.0, 20.0],
        "zdelta": [5.0],
        "tdelta": [2 * np.pi],
        "depth": 1000.0,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        BaseClass(**kwargs)


@pytest.mark.parametrize("radii", [[20.0, 10.0], [10.0, 10.0]])
def test_cylindrical_base_rejects_non_increasing_radii(radii):
    with pytest.raises(ValueError, match="strictly increasing"):
        BaseClass(radii=radii, zdelta=[5.0])


def test_cylindrical_base_accepts_theta_partition_covering_full_circle():
    grid = BaseClass(
        radii=[10.0, 20.0],
        zdelta=[5.0],
        tdelta=[np.pi, np.pi / 2, np.pi / 2],
    )

    np.testing.assert_allclose(grid.tdelta, [np.pi, np.pi / 2, np.pi / 2])


@pytest.mark.parametrize("tdelta", [[np.pi], [np.pi, np.pi / 2]])
def test_cylindrical_base_rejects_theta_partition_not_covering_full_circle(tdelta):
    with pytest.raises(ValueError, match="2\\*pi"):
        BaseClass(radii=[10.0, 20.0], zdelta=[5.0], tdelta=tdelta)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([], "cannot be empty"),
        ([1000.0, np.nan], "scalar"),
        ([1000.0, np.inf], "scalar"),
        (["bad"], "convertible"),
        ([[1000.0, 1005.0]], "scalar"),
    ],
)
def test_cylindrical_base_rejects_invalid_depth(value, message):
    with pytest.raises(ValueError, match=message):
        BaseClass(radii=[10.0, 20.0], zdelta=[5.0], depth=value)


def test_cylindrical_base_getters_do_not_expose_internal_arrays():
    grid = BaseClass(
        radii=[10.0, 20.0],
        zdelta=[5.0, 15.0],
        tdelta=[np.pi, np.pi],
    )

    radii = grid.radii
    zdelta = grid.zdelta
    tdelta = grid.tdelta

    radii[0] = 999.0
    zdelta[0] = 999.0
    tdelta[0] = 999.0

    np.testing.assert_allclose(grid.radii, [10.0, 20.0])
    np.testing.assert_allclose(grid.zdelta, [5.0, 15.0])
    np.testing.assert_allclose(grid.tdelta, [np.pi, np.pi])


class CachedCylindricalGeometry(BaseClass):
    @cached_property
    def geometry_signature(self):
        return (
            tuple(self.radii),
            tuple(self.zdelta),
            tuple(self.tdelta),
            float(self.depth),
        )

    def _invalidate_spatial_cache(self) -> None:
        self.__dict__.pop("geometry_signature", None)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("radii", [10.0, 30.0]),
        ("zdelta", [10.0, 20.0]),
        ("tdelta", [np.pi, np.pi]),
        ("depth", 900.0),
    ],
)
def test_cylindrical_base_notifies_subclasses_when_spatial_inputs_change(field, value):
    grid = CachedCylindricalGeometry(
        radii=[10.0, 20.0],
        zdelta=[5.0],
        tdelta=[2 * np.pi],
        depth=1000.0,
    )
    before = grid.geometry_signature
    assert "geometry_signature" in grid.__dict__

    setattr(grid, field, value)

    assert "geometry_signature" not in grid.__dict__
    assert grid.geometry_signature != before


def test_radial_grids_initializes_base_geometry_from_generated_radii():
    grid = RadialGrids(
        0.25,
        10.0,
        1000.0,
        zdelta=[5.0, 15.0],
        tdelta=[np.pi, np.pi],
        depth=900.0,
        num=10,
    )

    assert grid.nums_tuple == (10, 2, 2)
    np.testing.assert_allclose(grid.radii[[0, 5, -1]], [0.25, 10.0, 1000.0])
    np.testing.assert_allclose(grid.zdelta, [5.0, 15.0])
    np.testing.assert_allclose(grid.tdelta, [np.pi, np.pi])
    np.testing.assert_allclose(grid.depth, 900.0)


def test_radial_grids_get_radii_accepts_sequence_or_separate_outer_radii():
    separate = RadialGrids.get_radii(0.25, 10.0, 1000.0, num=10)
    sequence = RadialGrids.get_radii(0.25, [10.0, 1000.0], num=10)

    np.testing.assert_allclose(separate, sequence)
    assert separate.size == 11
    np.testing.assert_allclose(separate[[0, 5, -1]], [0.25, 10.0, 1000.0])


def test_radial_grids_allocates_remaining_cells_by_largest_remainder():
    cells = RadialGrids._allocate_region_cells(
        np.array([2.8, 2.7, 2.5, 2.0]),
        num=10,
    )

    np.testing.assert_array_equal(cells, [3, 3, 2, 2])


@pytest.mark.parametrize(
    ("well_radius", "outer_radii", "num", "message"),
    [
        (np.nan, (10.0,), 5, "well_radius must be finite"),
        (0.0, (10.0,), 5, "well_radius must be positive"),
        (1.0, (), 5, "At least two radii"),
        (1.0, (np.nan,), 5, "finite"),
        (1.0, (np.inf,), 5, "finite"),
        (1.0, (1.0,), 5, "strictly increasing"),
        (1.0, (10.0, 100.0), 1, "num must be at least"),
        (1.0, (10.0,), 5.5, "positive integer"),
        (1.0, (10.0,), np.nan, "num must be finite"),
    ],
)
def test_radial_grids_get_radii_rejects_invalid_inputs(
    well_radius,
    outer_radii,
    num,
    message,
):
    with pytest.raises(ValueError, match=message):
        RadialGrids.get_radii(well_radius, *outer_radii, num=num)


def test_radial_grids_plot3d_geometry_uses_public_field_units():
    grid = RadialGrids(
        0.25,
        10.0,
        zdelta=[5.0, 15.0],
        tdelta=[np.pi, np.pi],
        depth=900.0,
        num=4,
    )

    radii, theta_edges, depth_edges, theta_curve = grid._plot3d_geometry(angular_resolution=12)

    np.testing.assert_allclose(radii[[0, -1]], [0.25, 10.0])
    np.testing.assert_allclose(theta_edges, [0.0, np.pi, 2 * np.pi])
    np.testing.assert_allclose(depth_edges, [900.0, 905.0, 920.0])
    assert theta_curve.shape == (13,)
    np.testing.assert_allclose(theta_curve[[0, -1]], [0.0, 2 * np.pi])


@pytest.mark.parametrize(
    ("angular_resolution", "message"),
    [
        (7, "at least 8"),
        (8.5, "integer"),
        (np.nan, "finite"),
        ([12], "scalar"),
    ],
)
def test_radial_grids_plot3d_geometry_rejects_invalid_angular_resolution(
    angular_resolution,
    message,
):
    grid = RadialGrids(0.25, 10.0, zdelta=[5.0], num=4)

    with pytest.raises(ValueError, match=message):
        grid._plot3d_geometry(angular_resolution=angular_resolution)


def test_radial_grids_plot3d_returns_plotly_figure_when_available():
    go = pytest.importorskip("plotly.graph_objects")
    grid = RadialGrids(
        0.25,
        10.0,
        100.0,
        zdelta=[5.0, 15.0],
        tdelta=[np.pi, np.pi / 2, np.pi / 2],
        num=8,
    )

    fig = grid.plot3d(show=False)

    assert isinstance(fig, go.Figure)
    assert fig.layout.scene.zaxis.autorange == "reversed"
    assert len(fig.data) == 5
    assert [trace.type for trace in fig.data] == [
        "surface",
        "surface",
        "surface",
        "surface",
        "scatter3d",
    ]

    with pytest.raises(ValueError, match="vertical_exaggeration"):
        grid.plot3d(show=False, vertical_exaggeration=0.0)


def test_radial_grids_plot3d_writes_html_when_path_is_given(monkeypatch):
    go = pytest.importorskip("plotly.graph_objects")
    grid = RadialGrids(0.25, 10.0, zdelta=[5.0], num=4)
    call = {}

    def fake_write_html(self, file, **kwargs):
        call["file"] = file
        call.update(kwargs)

    monkeypatch.setattr(go.Figure, "write_html", fake_write_html)

    grid.plot3d(html_path="radial-grid.html")

    assert call["file"] == "radial-grid.html"
    assert call["include_plotlyjs"] is True
    assert call["full_html"] is True
    assert call["auto_open"] is False
