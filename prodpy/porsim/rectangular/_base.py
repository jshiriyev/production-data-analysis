import numpy as np

class GridBase:
	"""
    Base container for rectangular grid lengths.

    Public length properties are expressed in oilfield units (ft). Internal
    underscored arrays are stored in SI units (m). Scalar and array-like inputs
    are converted to one-dimensional NumPy arrays.

    Parameters
    ----------
    xdelta : array-like of float
        Cell lengths in the x direction, ft.
    ydelta : array-like of float
        Cell lengths in the y direction, ft.
    zdelta : array-like of float
        Cell lengths in the z direction, ft.
    depths : array-like of float, default 1000.0
        Grid depths, ft.

	All inputs are flattened to 1-D NumPy arrays, so scalar input becomes shape (1,).

    Attributes
    ----------
    _xdelta, _ydelta, _zdelta, _depths : numpy.ndarray
        Internal SI values, m.
    xnums, ynums, znums : int
        Number of cells along each axis.

    """
	FEET_TO_METERS = 0.3048

	def __init__(
		self,
		xdelta: np.typing.ArrayLike,
		ydelta: np.typing.ArrayLike,
		zdelta: np.typing.ArrayLike,
		depths: np.typing.ArrayLike=1000.
	):
		"""
		Initialize grid cell dimensions in feet.

		Parameters
		----------
		xdelta (float or array-like): Grid cell size in the x-direction (feet).
		ydelta (float or array-like): Grid cell size in the y-direction (feet).
		zdelta (float or array-like): Grid cell size in the z-direction (feet).

		depths (float or array-like): Grid depths (feet).
		"""
		self.xdelta = xdelta # ft
		self.ydelta = ydelta # ft
		self.zdelta = zdelta # ft
		self.depths = depths # ft

	@classmethod
	def _to_si_length_array(
		cls,
		value: np.typing.ArrayLike,
		name: str,
		*,
		require_positive: bool = False
	) -> np.ndarray:
		"""Validate a length input in feet and return a 1-D SI array in meters."""
		try:
			array = np.ravel(value).astype(float)
		except (TypeError, ValueError) as error:
			raise ValueError(f"{name} must be convertible to a numeric array.") from error

		if array.size == 0:
			raise ValueError(f"{name} cannot be empty.")

		if not np.all(np.isfinite(array)):
			raise ValueError(f"{name} must contain only finite values.")

		if require_positive and np.any(array <= 0):
			raise ValueError(f"{name} must contain only positive values.")

		return array*cls.FEET_TO_METERS
		
	@property
	def xdelta(self):
		"""Returns x-direction grid cell size in feet."""
		return self._xdelta/self.FEET_TO_METERS

	@xdelta.setter
	def xdelta(self,value):
		"""Sets x-direction grid cell size after converting from feet to meters."""
		self._xdelta = self._to_si_length_array(value,"xdelta",require_positive=True)

	@property
	def ydelta(self):
		"""Returns y-direction grid cell size in feet."""
		return self._ydelta/self.FEET_TO_METERS

	@ydelta.setter
	def ydelta(self,value):
		"""Sets y-direction grid cell size after converting from feet to meters."""
		self._ydelta = self._to_si_length_array(value,"ydelta",require_positive=True)

	@property
	def zdelta(self):
		"""Returns z-direction grid cell size in feet."""
		return self._zdelta/self.FEET_TO_METERS

	@zdelta.setter
	def zdelta(self,value):
		"""Sets z-direction grid cell size after converting from feet to meters."""
		self._zdelta = self._to_si_length_array(value,"zdelta",require_positive=True)

	@property
	def depths(self):
		"""Returns the depths of grids in feet."""
		return self._depths/self.FEET_TO_METERS

	@depths.setter
	def depths(self,value):
		"""Sets the depths of grids after converting from feet to meters."""
		self._depths = self._to_si_length_array(value,"depths")

	def __repr__(self) -> str:
		return (
			f"{self.__class__.__name__}("
			f"xnums={self.xnums}, ynums={self.ynums}, znums={self.znums}, "
			f")"
		)

	@property
	def tnums(self) -> tuple[int, int, int]:
		"""Returns tuple of (xnums,ynums,znums)."""
		return (self.xnums,self.ynums,self.znums)

	@property
	def xnums(self) -> int:
		"""Returns the number of grids in x-direction."""
		return self._xdelta.size

	@property
	def ynums(self) -> int:
		"""Returns the number of grids in y-direction."""
		return self._ydelta.size

	@property
	def znums(self) -> int:
		"""Returns the number of grids in z-direction."""
		return self._zdelta.size

	def _plot3d_geometry(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""Return x, y, and depth-node arrays in oilfield units for 3-D plotting."""
		if hasattr(self,"table"):
			raise ValueError("plot3d expects axis-wise grid spacing, not flattened Grids data.")

		depths = self.depths

		if depths.size == 1:
			cell_depths = np.full((self.ynums,self.xnums),float(depths[0]))
		elif depths.size == self.xnums*self.ynums:
			cell_depths = depths.reshape(self.ynums,self.xnums)
		else:
			raise ValueError(
				"depths must be scalar or contain xnums*ynums values for 3-D plotting."
			)

		xedges = np.insert(np.cumsum(self.xdelta),0,0.)
		yedges = np.insert(np.cumsum(self.ydelta),0,0.)
		zedges = np.insert(np.cumsum(self.zdelta),0,0.)

		depth_sum = np.zeros((self.ynums+1,self.xnums+1),dtype=float)
		depth_count = np.zeros_like(depth_sum)

		depth_sum[:-1,:-1] += cell_depths
		depth_count[:-1,:-1] += 1
		depth_sum[1:,:-1] += cell_depths
		depth_count[1:,:-1] += 1
		depth_sum[:-1,1:] += cell_depths
		depth_count[:-1,1:] += 1
		depth_sum[1:,1:] += cell_depths
		depth_count[1:,1:] += 1

		top_depths = depth_sum/depth_count
		depth_layers = top_depths[None,:,:]+zedges[:,None,None]

		return xedges,yedges,depth_layers

	def plot3d(
		self,
		title: str | None = None,
		*,
		show: bool = False,
		show_surfaces: bool = True,
		show_edges: bool = True,
		surface_opacity: float = 0.32,
		edge_color: str = "rgba(15, 23, 42, 0.62)",
		colorscale: str = "Viridis",
		vertical_exaggeration: float = 5.0,
		html_path: str | None = None,
		include_plotlyjs: bool | str = True,
		width: int | None = None,
		height: int = 720
	):
		"""
		Build an interactive 3-D Plotly figure of the structured rectangular grid.

		The plot uses oilfield units on all public axes: x and y are in feet,
		and the vertical axis is depth in feet with increasing depth downward.
		`depths` must be either scalar or contain one value per x-y cell.

		Parameters
		----------
		title : str, optional
			Figure title. Defaults to a grid-size summary.
		show : bool, default False
			If True, immediately display the figure.
		show_surfaces : bool, default True
			If True, draw translucent top, bottom, and side surfaces.
		show_edges : bool, default True
			If True, draw grid-line edges for all cell boundaries.
		surface_opacity : float, default 0.32
			Opacity applied to the translucent surfaces.
		edge_color : str, default "rgba(15, 23, 42, 0.62)"
			Plotly color used for grid-line edges.
		colorscale : str, default "Viridis"
			Plotly colorscale used for depth-colored surfaces.
		vertical_exaggeration : float, default 5.0
			Visual z-axis exaggeration used only for the scene aspect ratio.
		html_path : str, optional
			Path to write a standalone HTML plot. If omitted and `show=True`, a
			temporary standalone HTML file is created and opened.
		include_plotlyjs : bool or str, default True
			Passed to Plotly's HTML writer. The default embeds Plotly so the
			browser page works offline.
		width : int, optional
			Figure width in pixels.
		height : int, default 720
			Figure height in pixels.

		Returns
		-------
		plotly.graph_objects.Figure
			Interactive 3-D grid figure.
		"""
		try:
			import plotly.graph_objects as go
		except ImportError as error:
			raise ImportError("3-D grid plotting requires plotly. Install prodpy[plots].") from error

		if vertical_exaggeration <= 0:
			raise ValueError("vertical_exaggeration must be positive.")

		xedges,yedges,depth_layers = self._plot3d_geometry()
		xmesh,ymesh = np.meshgrid(xedges,yedges)
		fig = go.Figure()

		if show_surfaces:
			surface_kwargs = {
				"colorscale": colorscale,
				"opacity": surface_opacity,
				"hovertemplate": "X %{x:.2f} ft<br>Y %{y:.2f} ft<br>Depth %{z:.2f} ft<extra></extra>",
				"contours": {"z": {"show": False}},
			}
			fig.add_trace(
				go.Surface(
					x=xmesh,
					y=ymesh,
					z=depth_layers[0],
					surfacecolor=depth_layers[0],
					name="Top depth",
					showscale=True,
					colorbar={"title": "Depth, ft", "len": 0.72},
					**surface_kwargs
				)
			)
			fig.add_trace(
				go.Surface(
					x=xmesh,
					y=ymesh,
					z=depth_layers[-1],
					surfacecolor=depth_layers[-1],
					name="Base depth",
					showscale=False,
					**surface_kwargs
				)
			)

			layer_edges = np.arange(depth_layers.shape[0])
			y_side = np.tile(yedges,(depth_layers.shape[0],1))
			x_side = np.tile(xedges,(depth_layers.shape[0],1))

			side_surfaces = (
				(np.full_like(y_side,xedges[0],dtype=float),y_side,depth_layers[:,:,0],"X-min side"),
				(np.full_like(y_side,xedges[-1],dtype=float),y_side,depth_layers[:,:,-1],"X-max side"),
				(x_side,np.full_like(x_side,yedges[0],dtype=float),depth_layers[:,0,:],"Y-min side"),
				(x_side,np.full_like(x_side,yedges[-1],dtype=float),depth_layers[:,-1,:],"Y-max side"),
			)

			for xside,yside,zside,name in side_surfaces:
				fig.add_trace(
					go.Surface(
						x=xside,
						y=yside,
						z=zside,
						surfacecolor=np.repeat(layer_edges[:,None],zside.shape[1],axis=1),
						name=name,
						showscale=False,
						**surface_kwargs
					)
				)

		if show_edges:
			xline = []
			yline = []
			zline = []

			for layer in depth_layers:
				for j,yedge in enumerate(yedges):
					xline.extend([*xedges,None])
					yline.extend([*np.full_like(xedges,yedge),None])
					zline.extend([*layer[j,:],None])

				for i,xedge in enumerate(xedges):
					xline.extend([*np.full_like(yedges,xedge),None])
					yline.extend([*yedges,None])
					zline.extend([*layer[:,i],None])

			for j,yedge in enumerate(yedges):
				for i,xedge in enumerate(xedges):
					xline.extend([*np.full(depth_layers.shape[0],xedge),None])
					yline.extend([*np.full(depth_layers.shape[0],yedge),None])
					zline.extend([*depth_layers[:,j,i],None])

			fig.add_trace(
				go.Scatter3d(
					x=xline,
					y=yline,
					z=zline,
					mode="lines",
					line={"color": edge_color, "width": 3},
					name="Grid lines",
					hoverinfo="skip"
				)
			)

		xspan = max(float(np.ptp(xedges)),1.)
		yspan = max(float(np.ptp(yedges)),1.)
		zspan = max(float(np.ptp(depth_layers))*vertical_exaggeration,1.)
		maxspan = max(xspan,yspan,zspan)

		fig.update_layout(
			title=(
				title or
				f"{self.__class__.__name__} 3-D Grid ({self.xnums} x {self.ynums} x {self.znums})"
			),
			template="plotly_white",
			width=width,
			height=height,
			margin={"l": 0, "r": 0, "t": 58, "b": 0},
			showlegend=False,
			scene={
				"xaxis": {
					"title": "X, ft",
					"backgroundcolor": "rgba(248, 250, 252, 0.72)",
					"gridcolor": "#d7dee8",
				},
				"yaxis": {
					"title": "Y, ft",
					"backgroundcolor": "rgba(248, 250, 252, 0.72)",
					"gridcolor": "#d7dee8",
				},
				"zaxis": {
					"title": "Depth, ft",
					"autorange": "reversed",
					"backgroundcolor": "rgba(248, 250, 252, 0.72)",
					"gridcolor": "#d7dee8",
				},
				"aspectmode": "manual",
				"aspectratio": {"x": xspan/maxspan, "y": yspan/maxspan, "z": zspan/maxspan},
				"camera": {"eye": {"x": 1.6, "y": -1.75, "z": 1.08}},
			}
		)

		if show or html_path is not None:
			if html_path is None:
				import tempfile

				with tempfile.NamedTemporaryFile(
					prefix="prodpy-grid-",
					suffix=".html",
					delete=False
				) as html_file:
					html_path = html_file.name

			fig.write_html(
				html_path,
				include_plotlyjs=include_plotlyjs,
				full_html=True,
				auto_open=show,
				config={"responsive": True, "displaylogo": False}
			)

		return fig
