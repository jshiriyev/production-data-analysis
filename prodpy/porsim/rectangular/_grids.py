import numpy as np

class Grids:
	"""
	Internal flattened grid container for reservoir simulation.

	`Grids` is produced by `RectGrids.grids` and expects already-expanded
	per-cell arrays in oilfield units plus a precomputed neighbor table. It keeps
	only lightweight internal shape checks; user-facing validation belongs to
	`RectGrids`.
	"""
	FEET_TO_METERS = 0.3048

	def __init__(
		self,
		xdelta: np.ndarray,
		ydelta: np.ndarray,
		zdelta: np.ndarray,
		depths: np.ndarray,
		table: np.ndarray,
	):
		"""
		Initialize flattened per-cell grid data.

		Parameters
		----------
		xdelta : np.ndarray
			Per-cell grid sizes in the x-direction (feet), shape = (nums,).
		ydelta : np.ndarray
			Per-cell grid sizes in the y-direction (feet), shape = (nums,).
		zdelta : np.ndarray
			Per-cell grid sizes in the z-direction (feet), shape = (nums,).
		depths : np.ndarray
			Per-cell top-face depths (feet), shape = (nums,).
		table : np.ndarray
			Neighbor table with shape = (nums, 2*dims), where dims is 1, 2, or 3.

		"""
		self._xdelta = self._to_si_units(xdelta)
		self._ydelta = self._to_si_units(ydelta)
		self._zdelta = self._to_si_units(zdelta)
		self._depths = self._to_si_units(depths)
		self._table = np.asarray(table,dtype=np.int_)

		self._check_internal_shapes()

	@classmethod
	def _to_si_units(cls, array: np.ndarray) -> np.ndarray:
		"""Convert an oilfield unit array in feet to SI units in meters."""
		return np.asarray(array,dtype=float) * cls.FEET_TO_METERS
	
	@classmethod
	def _to_field_units(cls, array: np.ndarray) -> np.ndarray:
		"""Convert an SI array in meters to oilfield units in feet."""
		return array / cls.FEET_TO_METERS

	def _check_internal_shapes(self) -> None:
		"""Check constructor-only invariants for flattened simulation arrays."""
		arrays = (self._xdelta,self._ydelta,self._zdelta,self._depths)
		if any(array.ndim != 1 for array in arrays):
			raise ValueError("Grids expects flattened one-dimensional arrays.")

		if len({array.shape for array in arrays}) != 1:
			raise ValueError("Grids arrays must have matching per-cell shapes.")

		if (
			self._table.ndim != 2
			or self._table.shape[0] != self._xdelta.size
			or self._table.shape[1] not in {2,4,6}
		):
			raise ValueError("table must have shape (nums, 2*dims) for dims 1, 2, or 3.")

	def __repr__(self) -> str:
		return f"{self.__class__.__name__}(nums={self.nums}, dims={self.dims})"

	@property
	def xdelta(self):
		"""Returns per-cell x-direction grid sizes in feet."""
		return self._to_field_units(self._xdelta)

	@property
	def ydelta(self):
		"""Returns per-cell y-direction grid sizes in feet."""
		return self._to_field_units(self._ydelta)

	@property
	def zdelta(self):
		"""Returns per-cell z-direction grid sizes in feet."""
		return self._to_field_units(self._zdelta)

	@property
	def depths(self):
		"""Returns per-cell top-face depths in feet."""
		return self._to_field_units(self._depths)

	@property
	def table(self):
		"""Returns the neighbor table for flattened grid cells."""
		return self._table

	@property
	def xarea(self):
		return self._xarea / self.FEET_TO_METERS**2

	@property
	def _xarea(self):
		return self._ydelta*self._zdelta

	@property
	def yarea(self):
		return self._yarea / self.FEET_TO_METERS**2

	@property
	def _yarea(self):
		return self._zdelta*self._xdelta

	@property
	def zarea(self):
		return self._zarea / self.FEET_TO_METERS**2

	@property
	def _zarea(self):
		return self._xdelta*self._ydelta
	
	@property
	def volume(self):
		"""Returns the volume of grids in field units."""
		return self._volume / self.FEET_TO_METERS**3

	@property
	def _volume(self):
		return np.prod((self._xdelta,self._ydelta,self._zdelta),axis=0)

	@property
	def nums(self):
		"""Returns total number of grids."""
		return self.table.shape[0]

	@property
	def dims(self):
		"""Returns the flow dimensions."""
		return int(self.table.shape[1]/2)

	@property
	def index(self):
		"""Returns the indices of grids."""
		return np.arange(self.nums)
	
	@property
	def delta(self):
		"""Returns the cell sizes in x, y, and z direction, shape = (nums,3)."""
		return np.column_stack(
			(self.xdelta, self.ydelta, self.zdelta)
		)

	@property
	def xmin(self):
		"""Returns x-minimum boundary indices."""
		return self.index[self._xmin]

	@property
	def _xmin(self):
		"""Returns x-minimum boundary boolean."""
		return self.index==self.table[:,0]

	@property
	def xpos(self):
		"""Returns x-positive neighbor indices."""
		return self.index[self._xpos]

	@property
	def _xpos(self):
		"""Returns x-positive neighbor boolean."""
		return self.index!=self.table[:,0]

	@property
	def xneg(self):
		"""Returns x-negative neighbor indices."""
		return self.index[self._xneg]

	@property
	def _xneg(self):
		"""Returns x-negative neighbor boolean."""
		return self.index!=self.table[:,1]

	@property
	def xmax(self):
		"""Returns x-maximum boundary indices."""
		return self.index[self._xmax]

	@property
	def _xmax(self):
		"""Returns x-maximum boundary boolean."""
		return self.index==self.table[:,1]

	@property
	def ymin(self):
		"""Returns y-minimum boundary indices."""
		return self.index[self._ymin]

	@property
	def _ymin(self):
		"""Returns y-minimum boundary boolean."""
		return self.index==self.table[:,2] if self.dims>1 else np.full(self.nums,True)

	@property
	def ypos(self):
		"""Returns y-positive neighbor indices."""
		return self.index[self._ypos]

	@property
	def _ypos(self):
		"""Returns y-positive neighbor boolean."""
		return self.index!=self.table[:,2] if self.dims>1 else np.full(self.nums,False)

	@property
	def yneg(self):
		"""Returns y-negative neighbor indices."""
		return self.index[self._yneg]

	@property
	def _yneg(self):
		"""Returns y-negative neighbor boolean."""
		return self.index!=self.table[:,3] if self.dims>1 else np.full(self.nums,False)

	@property
	def ymax(self):
		"""Returns y-maximum boundary indices."""
		return self.index[self._ymax]

	@property
	def _ymax(self):
		"""Returns y-maximum boundary boolean."""
		return self.index==self.table[:,3] if self.dims>1 else np.full(self.nums,True)

	@property
	def zmin(self):
		"""Returns z-minimum boundary indices."""
		return self.index[self._zmin]

	@property
	def _zmin(self):
		"""Returns z-minimum boundary boolean."""
		return self.index==self.table[:,4] if self.dims>2 else np.full(self.nums,True)

	@property
	def zpos(self):
		"""Returns z-positive neighbor indices."""
		return self.index[self._zpos]

	@property
	def _zpos(self):
		"""Returns z-positive neighbor boolean."""
		return self.index!=self.table[:,4] if self.dims>2 else np.full(self.nums,False)

	@property
	def zneg(self):
		"""Returns z-negative neighbor indices."""
		return self.index[self._zneg]

	@property
	def _zneg(self):
		"""Returns z-negative neighbor boolean."""
		return self.index!=self.table[:,5] if self.dims>2 else np.full(self.nums,False)

	@property
	def zmax(self):
		"""Returns z-maximum boundary indices."""
		return self.index[self._zmax]

	@property
	def _zmax(self):
		"""Returns z-maximum boundary boolean."""
		return self.index==self.table[:,5] if self.dims>2 else np.full(self.nums,True)
