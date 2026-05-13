from functools import cached_property

import numpy as np

from ._base_class import BaseClass
from ._grids import Grids

class RectGrids(BaseClass):
	"""Interface to get a three-dimensional rectangular cuboid for cell-based flow simulations."""
	
	def __init__(
		self,
		xdelta: np.ndarray,
		ydelta: np.ndarray,
		zdelta: np.ndarray,
		depths: float = 1000.,
		dims: int|None = None
	):
		"""
		Initialize a structured rectangular reservoir grid.
		
		Parameters
		----------
		xdelta : float or 1-D array-like of float
			Cell lengths in the x-direction, ft.
		ydelta : float or 1-D array-like of float
			Cell widths in the y-direction, ft.
		zdelta : float or 1-D array-like of float
			Cell heights in the z-direction, ft.
		depths : float, default 1000.0
			Depth of the reservoir-domain top surface, ft.
		dims   : {1, 2, 3}, optional
			Flow dimension used to build the neighbor table. If omitted, the
			dimension is inferred from active grid counts.
		"""
		super().__init__(xdelta,ydelta,zdelta,depths)

		self.dims = dims

	@property
	def length(self):
		"""Returns the size of reservoir domain in x-direction in feet."""
		return self._to_field_units(self._length)

	@property
	def _length(self):
		"""Calculates the size of reservoir domain in x-direction."""
		return self._xdelta.sum()

	@property
	def width(self):
		"""Returns the size of reservoir domain in y-direction in feet."""
		return self._to_field_units(self._width)

	@property
	def _width(self):
		"""Calculates the size of reservoir domain in y-direction."""
		return self._ydelta.sum()

	@property
	def height(self):
		"""Returns the size of reservoir domain in z-direction in feet."""
		return self._to_field_units(self._height)

	@property
	def _height(self):
		"""Calculates the size of reservoir domain in z-direction."""
		return self._zdelta.sum()

	def __repr__(self) -> str:
		return (
			f"{self.__class__.__name__}("
			f"xnums={self.xnums}, ynums={self.ynums}, znums={self.znums}"
			f")"
		)

	@property
	def nums_tuple(self) -> tuple[int, int, int]:
		"""Returns tuple of (xnums,ynums,znums)."""
		return (self.xnums, self.ynums, self.znums)

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
	
	@property
	def dims(self):
		"""Returns the flow dimensions."""
		return self._dims
	
	def _infer_dims(self) -> int:
		"""Infer flow dimensions from active grid counts."""
		if self.nums_tuple[2]>1:
			return 3
		if self.nums_tuple[1]>1:
			return 2
		return 1

	@dims.setter
	def dims(self,value):
		"""Calculates the flow dimensions."""
		inferred_dims = self._infer_dims()
		if value is not None:
			if value not in {1,2,3}:
				raise ValueError("dims must be 1, 2, 3, or None.")
			if value < inferred_dims:
				raise ValueError(
					f"dims={value} is too low for the number of active grids. "
					f"At least {inferred_dims} dimensions are required for nums={self.nums_tuple}."
				)
			self._dims_inferred = False
			self._dims = value
		else:
			self._dims_inferred = True
			self._dims = inferred_dims

	def _invalidate_spatial_cache(self) -> None:
		"""Clear cached geometry derived from spatial arrays."""
		self.__dict__.pop("table",None)
		self.__dict__.pop("grids",None)

		if getattr(self,"_dims_inferred",False):
			self._dims = self._infer_dims()

	@staticmethod
	def get_interval_midpoints(array:np.ndarray):
		"""Calculates the midpoints of the given array."""
		value = np.cumsum(array)
		return (np.insert(value[:-1],0,0)+value)/2
	
	@property
	def xcenter(self):
		"""Returns the x-center of the grids, shape = (xnums,)."""
		return self._to_field_units(self._xcenter)

	@property
	def _xcenter(self):
		"""Calculates the x-center of the grids, shape = (xnums,)."""
		return self.get_interval_midpoints(self._xdelta)

	@property
	def ycenter(self):
		"""Returns the y-center of the grids, shape = (ynums,)."""
		return self._to_field_units(self._ycenter)

	@property
	def _ycenter(self):
		"""Calculates the y-center of the grids, shape = (ynums,)."""
		return self.get_interval_midpoints(self._ydelta)

	@property
	def zcenter(self):
		"""Returns the z-center of the grids, shape = (znums,)."""
		return self._to_field_units(self._zcenter)

	@property
	def _zcenter(self):
		"""Calculates the z-center of the grids, shape = (znums,)."""
		return self.get_interval_midpoints(self._zdelta)

	@property
	def index(self):
		"""Returns the indices of all grids."""
		return np.arange(np.prod(self.nums_tuple),dtype=np.int_)
	
	@classmethod
	def from_delta_arrays(
		cls,
		xdelta: np.ndarray,
		ydelta: np.ndarray,
		zdelta: np.ndarray,
		depths: float = 1000.,
		dims: int|None = None
	):
		"""
		Create a rectangular grid from explicit axis cell-size arrays.
		
		Parameters
		----------
		xdelta : float or 1-D array-like of float
			Cell lengths in the x-direction, ft.
		ydelta : float or 1-D array-like of float
			Cell widths in the y-direction, ft.
		zdelta : float or 1-D array-like of float
			Cell heights in the z-direction, ft.
		depths : float, default 1000.0
			Depth of the reservoir-domain top surface, ft.
		dims   : {1, 2, 3}, optional
			Flow dimension used to build the neighbor table.
		"""
		return cls(xdelta,ydelta,zdelta,depths,dims)

	@staticmethod
	def _validate_size_tuple(size: tuple[float,float,float]) -> tuple[float,float,float]:
		"""Return a validated positive three-value reservoir size tuple."""
		try:
			array = np.asarray(size,dtype=float)
		except (TypeError, ValueError) as error:
			raise ValueError("size must contain three numeric values.") from error

		if array.shape != (3,):
			raise ValueError("size must contain exactly three values.")

		if not np.all(np.isfinite(array)):
			raise ValueError("size must contain only finite values.")

		if np.any(array <= 0):
			raise ValueError("size must contain only positive values.")

		return tuple(float(value) for value in array)

	@staticmethod
	def _validate_nums_tuple(nums: tuple[int,int,int]) -> tuple[int,int,int]:
		"""Return a validated positive three-value grid-count tuple."""
		try:
			array = np.asarray(nums,dtype=float)
		except (TypeError, ValueError) as error:
			raise ValueError("nums must contain three integer values.") from error

		if array.shape != (3,):
			raise ValueError("nums must contain exactly three values.")

		if not np.all(np.isfinite(array)):
			raise ValueError("nums must contain only finite values.")

		if np.any(array <= 0) or not np.all(array == np.floor(array)):
			raise ValueError("nums must contain only positive integers.")

		return tuple(int(value) for value in array)

	@classmethod
	def from_size_and_nums(
		cls,
		size: tuple[float,float,float],
		nums: tuple[int,int,int],
		depths: float = 1000.,
		dims: int|None =None
	):
		"""
		Create a rectangular grid from reservoir size and cell counts.

		Parameters
		----------
		size : tuple of float
			Reservoir dimensions ``(length, width, height)`` in feet.
		nums : tuple of int
			Number of cells along ``(x, y, z)``.
		depths : float, default 1000.0
			Depth of the reservoir-domain top surface, ft.
		dims : {1, 2, 3}, optional
			Flow dimension used to build the neighbor table.
		"""
		size = cls._validate_size_tuple(size)
		nums = cls._validate_nums_tuple(nums)

		xdelta = np.full(nums[0],size[0]/nums[0])
		ydelta = np.full(nums[1],size[1]/nums[1])
		zdelta = np.full(nums[2],size[2]/nums[2])

		return cls(xdelta,ydelta,zdelta,depths,dims)

	@cached_property
	def grids(self):
		"""Returns Grids instance necessary for flow calculations."""
		xynums = self.xnums*self.ynums # number of grids in a x-y plane
		yznums = self.ynums*self.znums # number of grids in a y-z plane

		xdelta = np.tile(self.xdelta,yznums)
		ydelta = np.repeat(self.ydelta,self.xnums)
		ydelta = np.tile(ydelta,self.znums)
		zdelta = np.repeat(self.zdelta,xynums)

		depths = np.full(xynums,float(self.depths[0]))

		height = np.cumsum(np.insert(self.zdelta[:-1],0,0))
		depths = np.tile(depths,self.znums)+np.repeat(height,xynums)

		return Grids(xdelta,ydelta,zdelta,depths,self.table)
	
	@cached_property
	def table(self):
		"""Returns the table of grids that stores neighborhood indices."""
		plat = np.tile(self.index,(self.dims*2,1)).T

		plat[self.index.reshape(-1,self.xnums)[:,1:].ravel(),0] -= 1
		plat[self.index.reshape(-1,self.xnums)[:,:-1].ravel(),1] += 1

		if self.dims>1:
			plat[self.index.reshape(self.znums,-1)[:,self.xnums:],2] -= self.xnums
			plat[self.index.reshape(self.znums,-1)[:,:-self.xnums],3] += self.xnums

		if self.dims>2:
			plat[self.index.reshape(self.znums,-1)[1:,:],4] -= self.xnums*self.ynums
			plat[self.index.reshape(self.znums,-1)[:-1,:],5] += self.xnums*self.ynums

		return plat
	
	def _plot3d_geometry(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""Return x, y, and depth-node arrays in oilfield units for 3-D plotting."""
		xedges = np.insert(np.cumsum(self.xdelta),0,0.)
		yedges = np.insert(np.cumsum(self.ydelta),0,0.)
		zedges = np.insert(np.cumsum(self.zdelta),0,0.)

		top_depths = np.full((self.ynums+1,self.xnums+1),float(self.depths[0]))
		depth_layers = top_depths[None,:,:]+zedges[:,None,None]

		return xedges, yedges, depth_layers

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
		The top surface is flat at `depths`.

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
