import numpy as np
import pytest

from prodpy.porsim.rectangular._base_class import BaseClass
from prodpy.porsim.rectangular._rect_grids import RectGrids


def test_grid_base_converts_public_feet_to_internal_si_arrays():
	grid = BaseClass([10.0, 20.0], 30.0, [40.0], depths=0.0)

	np.testing.assert_allclose(grid.xdelta, [10.0, 20.0])
	np.testing.assert_allclose(grid._xdelta, np.array([10.0, 20.0])*BaseClass.FEET_TO_METERS)
	np.testing.assert_allclose(grid.ydelta, [30.0])
	np.testing.assert_allclose(grid.zdelta, [40.0])
	np.testing.assert_allclose(grid.depths, [0.0])


@pytest.mark.parametrize("field", ["xdelta", "ydelta", "zdelta"])
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
def test_grid_base_rejects_invalid_cell_dimensions(field, value, message):
	kwargs = {
		"xdelta": [10.0],
		"ydelta": [20.0],
		"zdelta": [30.0],
		"depths": 1000.0,
	}
	kwargs[field] = value

	with pytest.raises(ValueError, match=message):
		BaseClass(**kwargs)


@pytest.mark.parametrize(
	("value", "message"),
	[
		([], "scalar"),
		([1000.0, np.nan], "scalar"),
		([1000.0, np.inf], "scalar"),
		(["bad"], "convertible"),
		([[1000.0, 1005.0]], "scalar"),
	],
)
def test_grid_base_rejects_invalid_depths(value, message):
	with pytest.raises(ValueError, match=message):
		BaseClass([10.0], [20.0], [30.0], depths=value)


def test_rect_grids_plot3d_geometry_uses_public_oilfield_units():
	grid = RectGrids([10.0, 20.0], [30.0], [5.0, 15.0], depths=1000.0)

	xedges,yedges,depth_layers = grid._plot3d_geometry()

	np.testing.assert_allclose(xedges, [0.0, 10.0, 30.0])
	np.testing.assert_allclose(yedges, [0.0, 30.0])
	assert depth_layers.shape == (3,2,3)
	np.testing.assert_allclose(
		depth_layers[0],
		[
			[1000.0, 1000.0, 1000.0],
			[1000.0, 1000.0, 1000.0],
		]
	)
	np.testing.assert_allclose(depth_layers[-1],depth_layers[0]+20.0)


def test_rect_grids_rejects_non_scalar_depths():
	with pytest.raises(ValueError, match="depths must be scalar"):
		RectGrids([10.0, 20.0], [30.0], [5.0], depths=[1000.0, 1005.0, 1010.0])


def test_rect_grids_plot3d_returns_plotly_figure_when_available():
	go = pytest.importorskip("plotly.graph_objects")
	grid = RectGrids([10.0], [20.0], [5.0], depths=1000.0)

	fig = grid.plot3d(show=False)

	assert isinstance(fig,go.Figure)
	assert fig.layout.scene.zaxis.autorange == "reversed"
	assert len(fig.data) >= 1

	with pytest.raises(ValueError, match="vertical_exaggeration"):
		grid.plot3d(show=False,vertical_exaggeration=0.0)


def test_rect_grids_plot3d_writes_html_when_path_is_given(monkeypatch):
	go = pytest.importorskip("plotly.graph_objects")
	grid = RectGrids([10.0], [20.0], [5.0], depths=1000.0)
	call = {}

	def fake_write_html(self, file, **kwargs):
		call["file"] = file
		call.update(kwargs)

	monkeypatch.setattr(go.Figure,"write_html",fake_write_html)

	grid.plot3d(html_path="grid.html")

	assert call["file"] == "grid.html"
	assert call["include_plotlyjs"] is True
	assert call["full_html"] is True
	assert call["auto_open"] is False


def test_rect_grids_clears_cached_geometry_when_dimensions_change():
	grid = RectGrids([10.0, 20.0], [30.0], [5.0], depths=1000.0)
	assert grid.table.shape == (2,2)
	assert grid.grids.nums == 2

	grid.xdelta = [10.0, 20.0, 30.0]

	assert "table" not in grid.__dict__
	assert "grids" not in grid.__dict__
	assert grid.table.shape == (3,2)
	assert grid.grids.nums == 3


def test_rect_grids_updates_inferred_dims_when_dimensions_change():
	grid = RectGrids([10.0], [20.0], [5.0], depths=1000.0)
	assert grid.dims == 1
	assert grid.table.shape == (1,2)

	grid.zdelta = [5.0, 5.0]

	assert grid.dims == 3
	assert grid.table.shape == (2,6)


def test_rect_grids_preserves_explicit_dims_when_dimensions_change():
	grid = RectGrids([10.0], [20.0], [5.0], depths=1000.0, dims=1)
	assert grid.dims == 1

	grid.zdelta = [5.0, 5.0]

	assert grid.dims == 1
	assert grid.table.shape == (2,2)


def test_rect_grids_allows_explicit_three_dimensional_flow_with_one_z_layer():
	grid = RectGrids([10.0], [20.0], [5.0], depths=1000.0, dims=3)

	assert grid.dims == 3
	assert grid.table.shape == (1,6)
	np.testing.assert_array_equal(grid.table, [[0,0,0,0,0,0]])


def test_rect_grids_from_size_and_nums_builds_uniform_grid():
	grid = RectGrids.from_size_and_nums((100.0, 80.0, 20.0), (2, 4, 1), depths=900.0)

	np.testing.assert_allclose(grid.xdelta, [50.0, 50.0])
	np.testing.assert_allclose(grid.ydelta, [20.0, 20.0, 20.0, 20.0])
	np.testing.assert_allclose(grid.zdelta, [20.0])
	np.testing.assert_allclose(grid.depths, [900.0])
	assert grid.dims == 2


@pytest.mark.parametrize(
	("size", "message"),
	[
		((100.0, 80.0), "exactly three"),
		((100.0, np.nan, 20.0), "finite"),
		((100.0, 0.0, 20.0), "positive"),
		(("bad", 80.0, 20.0), "numeric"),
	],
)
def test_rect_grids_from_size_and_nums_rejects_invalid_size(size, message):
	with pytest.raises(ValueError, match=message):
		RectGrids.from_size_and_nums(size, (2, 4, 1))


@pytest.mark.parametrize(
	("nums", "message"),
	[
		((2, 4), "exactly three"),
		((2, np.nan, 1), "finite"),
		((2, 0, 1), "positive integers"),
		((2, 1.5, 1), "positive integers"),
		(("bad", 4, 1), "integer"),
	],
)
def test_rect_grids_from_size_and_nums_rejects_invalid_nums(nums, message):
	with pytest.raises(ValueError, match=message):
		RectGrids.from_size_and_nums((100.0, 80.0, 20.0), nums)


def test_rect_grids_grids_expands_axis_arrays_to_flattened_cells():
	grids = RectGrids(
		[10.0, 20.0],
		[30.0, 40.0],
		[5.0, 15.0],
		depths=1000.0,
	).grids

	np.testing.assert_allclose(grids.xdelta, [10.0, 20.0, 10.0, 20.0, 10.0, 20.0, 10.0, 20.0])
	np.testing.assert_allclose(grids.ydelta, [30.0, 30.0, 40.0, 40.0, 30.0, 30.0, 40.0, 40.0])
	np.testing.assert_allclose(grids.zdelta, [5.0, 5.0, 5.0, 5.0, 15.0, 15.0, 15.0, 15.0])
	np.testing.assert_allclose(
		grids.depths,
		[1000.0, 1000.0, 1000.0, 1000.0, 1005.0, 1005.0, 1005.0, 1005.0],
	)
	assert grids.nums == 8
	assert grids.dims == 3


def test_rect_grids_grids_repr_summarizes_shape():
	grids = RectGrids([10.0, 20.0], [30.0], [5.0], depths=1000.0).grids

	assert repr(grids) == "Grids(nums=2, dims=1)"


def test_rect_grids_grids_preserves_public_field_units_and_internal_si_units():
	grids = RectGrids([10.0, 20.0], [30.0], [5.0], depths=1000.0).grids

	np.testing.assert_allclose(grids.xdelta, [10.0, 20.0])
	np.testing.assert_allclose(grids._xdelta, np.array([10.0, 20.0])*grids.FEET_TO_METERS)
	np.testing.assert_allclose(grids.ydelta, [30.0, 30.0])
	np.testing.assert_allclose(grids.zdelta, [5.0, 5.0])
	np.testing.assert_allclose(grids.depths, [1000.0, 1000.0])


def test_rect_grids_grids_calculates_delta_area_and_volume():
	grids = RectGrids([10.0, 20.0], [30.0], [5.0], depths=1000.0).grids

	np.testing.assert_allclose(grids.delta, [[10.0, 30.0, 5.0], [20.0, 30.0, 5.0]])
	np.testing.assert_allclose(grids.xarea, [150.0, 150.0])
	np.testing.assert_allclose(grids.yarea, [50.0, 100.0])
	np.testing.assert_allclose(grids.zarea, [300.0, 600.0])
	np.testing.assert_allclose(grids.volume, [1500.0, 3000.0])


def test_rect_grids_grids_exposes_neighbor_table_and_boundary_indices():
	grids = RectGrids(
		[10.0, 20.0],
		[30.0, 40.0],
		[5.0, 15.0],
		depths=1000.0,
	).grids

	np.testing.assert_array_equal(
		grids.table,
		[
			[0, 1, 0, 2, 0, 4],
			[0, 1, 1, 3, 1, 5],
			[2, 3, 0, 2, 2, 6],
			[2, 3, 1, 3, 3, 7],
			[4, 5, 4, 6, 0, 4],
			[4, 5, 5, 7, 1, 5],
			[6, 7, 4, 6, 2, 6],
			[6, 7, 5, 7, 3, 7],
		],
	)
	np.testing.assert_array_equal(grids.index, np.arange(8))
	np.testing.assert_array_equal(grids.xmin, [0, 2, 4, 6])
	np.testing.assert_array_equal(grids.xmax, [1, 3, 5, 7])
	np.testing.assert_array_equal(grids.ymin, [0, 1, 4, 5])
	np.testing.assert_array_equal(grids.ymax, [2, 3, 6, 7])
	np.testing.assert_array_equal(grids.zmin, [0, 1, 2, 3])
	np.testing.assert_array_equal(grids.zmax, [4, 5, 6, 7])
	np.testing.assert_array_equal(grids.xpos, [1, 3, 5, 7])
	np.testing.assert_array_equal(grids.xneg, [0, 2, 4, 6])
	np.testing.assert_array_equal(grids.ypos, [2, 3, 6, 7])
	np.testing.assert_array_equal(grids.yneg, [0, 1, 4, 5])
	np.testing.assert_array_equal(grids.zpos, [4, 5, 6, 7])
	np.testing.assert_array_equal(grids.zneg, [0, 1, 2, 3])


def test_rect_grids_grids_uses_boundary_defaults_for_inactive_dimensions():
	grids = RectGrids([10.0, 20.0], [30.0], [5.0], depths=1000.0).grids

	np.testing.assert_array_equal(grids.xmin, [0])
	np.testing.assert_array_equal(grids.xmax, [1])
	np.testing.assert_array_equal(grids.ymin, [0, 1])
	np.testing.assert_array_equal(grids.ymax, [0, 1])
	np.testing.assert_array_equal(grids.zmin, [0, 1])
	np.testing.assert_array_equal(grids.zmax, [0, 1])
	assert grids.ypos.size == 0
	assert grids.yneg.size == 0
	assert grids.zpos.size == 0
	assert grids.zneg.size == 0


def test_rect_grids_grids_public_fields_are_read_only():
	grids = RectGrids([10.0], [20.0], [5.0], depths=1000.0).grids

	with pytest.raises(AttributeError):
		grids.xdelta = [1.0]
	with pytest.raises(AttributeError):
		grids.ydelta = [1.0]
	with pytest.raises(AttributeError):
		grids.zdelta = [1.0]
	with pytest.raises(AttributeError):
		grids.depths = [1.0]
	with pytest.raises(AttributeError):
		grids.table = np.array([[0,1]])
