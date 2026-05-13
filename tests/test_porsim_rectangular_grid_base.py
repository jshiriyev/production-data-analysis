import numpy as np
import pytest

from prodpy.porsim.rectangular import GridBase


def test_grid_base_converts_public_feet_to_internal_si_arrays():
	grid = GridBase([[10.0, 20.0]], 30.0, [40.0], depths=0.0)

	np.testing.assert_allclose(grid.xdelta, [10.0, 20.0])
	np.testing.assert_allclose(grid._xdelta, np.array([10.0, 20.0])*GridBase.FEET_TO_METERS)
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
	],
)
def test_grid_base_rejects_invalid_cell_dimensions(field, value, message):
	kwargs = {
		"xdelta": [10.0],
		"ydelta": [20.0],
		"zdelta": [30.0],
	}
	kwargs[field] = value

	with pytest.raises(ValueError, match=message):
		GridBase(**kwargs)


@pytest.mark.parametrize(
	("value", "message"),
	[
		([], "cannot be empty"),
		([1000.0, np.nan], "finite"),
		([1000.0, np.inf], "finite"),
		(["bad"], "convertible"),
	],
)
def test_grid_base_rejects_invalid_depths(value, message):
	with pytest.raises(ValueError, match=message):
		GridBase([10.0], [20.0], [30.0], depths=value)


def test_grid_base_plot3d_geometry_uses_public_oilfield_units():
	grid = GridBase([10.0, 20.0], [30.0], [5.0, 15.0], depths=[1000.0, 1010.0])

	xedges,yedges,depth_layers = grid._plot3d_geometry()

	np.testing.assert_allclose(xedges, [0.0, 10.0, 30.0])
	np.testing.assert_allclose(yedges, [0.0, 30.0])
	assert depth_layers.shape == (3,2,3)
	np.testing.assert_allclose(
		depth_layers[0],
		[
			[1000.0, 1005.0, 1010.0],
			[1000.0, 1005.0, 1010.0],
		]
	)
	np.testing.assert_allclose(depth_layers[-1],depth_layers[0]+20.0)


def test_grid_base_plot3d_rejects_unplottable_depth_shape():
	grid = GridBase([10.0, 20.0], [30.0], [5.0], depths=[1000.0, 1005.0, 1010.0])

	with pytest.raises(ValueError, match="depths must be scalar"):
		grid._plot3d_geometry()


def test_grid_base_plot3d_rejects_flattened_grids_data():
	grid = GridBase([10.0], [20.0], [5.0], depths=1000.0)
	grid.table = np.array([[0,0]])

	with pytest.raises(ValueError, match="axis-wise grid spacing"):
		grid._plot3d_geometry()


def test_grid_base_plot3d_returns_plotly_figure_when_available():
	go = pytest.importorskip("plotly.graph_objects")
	grid = GridBase([10.0], [20.0], [5.0], depths=1000.0)

	fig = grid.plot3d(show=False)

	assert isinstance(fig,go.Figure)
	assert fig.layout.scene.zaxis.autorange == "reversed"
	assert len(fig.data) >= 1

	with pytest.raises(ValueError, match="vertical_exaggeration"):
		grid.plot3d(show=False,vertical_exaggeration=0.0)


def test_grid_base_plot3d_writes_standalone_html(tmp_path):
	pytest.importorskip("plotly.graph_objects")
	grid = GridBase([10.0], [20.0], [5.0], depths=1000.0)
	html_path = tmp_path/"grid.html"

	grid.plot3d(html_path=str(html_path))

	content = html_path.read_text(encoding="utf-8")
	assert "Plotly.newPlot" in content
	assert "plotly" in content.lower()


def test_grid_base_plot3d_show_uses_standalone_html(monkeypatch):
	go = pytest.importorskip("plotly.graph_objects")
	grid = GridBase([10.0], [20.0], [5.0], depths=1000.0)
	call = {}

	def fake_write_html(self, file, **kwargs):
		call["file"] = file
		call.update(kwargs)

	monkeypatch.setattr(go.Figure,"write_html",fake_write_html)

	grid.plot3d(show=True)

	assert call["file"].endswith(".html")
	assert call["include_plotlyjs"] is True
	assert call["full_html"] is True
	assert call["auto_open"] is True
