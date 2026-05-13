import numpy as np

class BaseClass:
	"""
	Base class container for rectangular grids. The aim is to provide a common base
	for different subclasses while ensuring consistent handling of spatial properties.

	Public length properties are expressed in oilfield units (ft). Internal
	underscored arrays are stored in SI units (m). Axis-spacing inputs must be
	scalar or one-dimensional and are stored internally as one-dimensional NumPy
	arrays.

	Parameters
	----------
	xdelta : float or 1-D array-like of float
		Cell lengths in the x direction, ft.
	ydelta : float or 1-D array-like of float
		Cell lengths in the y direction, ft.
	zdelta : float or 1-D array-like of float
		Cell lengths in the z direction, ft.
	depths : float
		Depth of the reservoir-domain top surface, ft.

	Attributes
	----------
	_xdelta, _ydelta, _zdelta, _depths : numpy.ndarray
		Internal SI values, m.

	"""
	FEET_TO_METERS = 0.3048

	def __init__(
		self,
		xdelta: np.typing.ArrayLike,
		ydelta: np.typing.ArrayLike,
		zdelta: np.typing.ArrayLike,
		depths: np.typing.ArrayLike,
	):
		self.xdelta = xdelta
		self.ydelta = ydelta
		self.zdelta = zdelta
		self.depths = depths

	@classmethod
	def _to_si_axis_array(
		cls,
		value: np.typing.ArrayLike,
		name: str,
		*,
		require_positive: bool = False
	) -> np.ndarray:
		"""Validate a length input in feet and return a 1-D SI array in meters."""
		try:
			array = np.asarray(value, dtype=float)
		except (TypeError, ValueError) as error:
			raise ValueError(f"{name} must be convertible to a numeric array.") from error

		if array.size == 0:
			raise ValueError(f"{name} cannot be empty.")

		if array.ndim == 0:
			array = array.reshape(1)
		elif array.ndim != 1:
			raise ValueError(f"{name} must be scalar or one-dimensional.")

		if not np.all(np.isfinite(array)):
			raise ValueError(f"{name} must contain only finite values.")

		if require_positive and np.any(array <= 0):
			raise ValueError(f"{name} must contain only positive values.")

		return array*cls.FEET_TO_METERS

	@classmethod
	def _to_si_scalar_length(
		cls,
		value: np.typing.ArrayLike,
		name: str
	) -> np.ndarray:
		"""Validate a scalar length input in feet and return a 1-D SI array in meters."""
		try:
			array = np.asarray(value,dtype=float)
		except (TypeError, ValueError) as error:
			raise ValueError(f"{name} must be convertible to a numeric scalar.") from error

		if array.ndim != 0:
			raise ValueError(f"{name} must be scalar.")

		if not np.isfinite(array):
			raise ValueError(f"{name} must be finite.")

		return array.reshape(1)*cls.FEET_TO_METERS
	
	@classmethod
	def _to_field_units(
		cls,
		array: np.ndarray
	) -> np.ndarray:
		"""Convert an SI array in meters to oilfield units in feet."""
		return array/cls.FEET_TO_METERS
	
	def _invalidate_spatial_cache(self) -> None:
		"""Clear subclass caches derived from spatial arrays."""
		return None
		
	@property
	def xdelta(self):
		"""Returns x-direction grid cell size in feet."""
		return self._to_field_units(self._xdelta)

	@xdelta.setter
	def xdelta(self,value):
		"""Sets x-direction grid cell size after converting from feet to meters."""
		self._xdelta = self._to_si_axis_array(value,"xdelta",require_positive=True)
		self._invalidate_spatial_cache()

	@property
	def ydelta(self):
		"""Returns y-direction grid cell size in feet."""
		return self._to_field_units(self._ydelta)

	@ydelta.setter
	def ydelta(self,value):
		"""Sets y-direction grid cell size after converting from feet to meters."""
		self._ydelta = self._to_si_axis_array(value,"ydelta",require_positive=True)
		self._invalidate_spatial_cache()

	@property
	def zdelta(self):
		"""Returns z-direction grid cell size in feet."""
		return self._to_field_units(self._zdelta)

	@zdelta.setter
	def zdelta(self,value):
		"""Sets z-direction grid cell size after converting from feet to meters."""
		self._zdelta = self._to_si_axis_array(value,"zdelta",require_positive=True)
		self._invalidate_spatial_cache()

	@property
	def depths(self):
		"""Returns the reservoir-domain top-surface depth in feet."""
		return self._to_field_units(self._depths)

	@depths.setter
	def depths(self,value):
		"""Sets reservoir-domain top-surface depth after converting feet to meters."""
		self._depths = self._to_si_scalar_length(value,"depths")
		self._invalidate_spatial_cache()
