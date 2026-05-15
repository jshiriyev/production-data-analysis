import numpy as np

class BaseClass:
    """
    Base container for cylindrical-grid geometry.

    Public length properties are expressed in oilfield units (ft). Internal
    underscored length arrays are stored in SI units (m). Angular spacing is
    expressed in radians and stored without unit conversion. Scalar and
    one-dimensional spacing inputs are normalized to zero and one-dimensional
    NumPy arrays, respectively.

    Parameters
    ----------
    radii : float or 1-D array-like of float
        Radial coordinates, ft. Values must be positive and, when multiple
        values are supplied, strictly increasing.
    zdelta : float or 1-D array-like of float
        Cell thicknesses in the z direction, ft. Values must be positive.
    tdelta : float or 1-D array-like of float, default 2*pi
        Cell angles in the theta direction, rad. Values must be positive.
    depth : float, default 1000.0
        Depth of the reservoir-domain top surface, ft.

    Attributes
    ----------
    _radii, _zdelta, _depth : numpy.ndarray
        Internal SI values, m.
    _tdelta : numpy.ndarray
        Internal angular spacing values, rad.

    Notes
    -----
    Axis-like public properties return one-dimensional NumPy arrays even when
    scalar inputs are supplied. ``depth`` is stored and returned as a scalar value.

    """
    FEET_TO_METERS = 0.3048

    def __init__(
        self,
        radii: np.typing.ArrayLike,
        zdelta: np.typing.ArrayLike,
        tdelta: np.typing.ArrayLike | None = None,
        depth: float = 1000.,
    ):
        self.radii = radii
        self.zdelta = zdelta
        self.tdelta = np.array([2*np.pi]) if tdelta is None else tdelta
        self.depth = depth

    @classmethod
    def _to_positive_array(
        cls,
        value: np.typing.ArrayLike,
        name: str,
    ) -> np.ndarray:
        """Validate a positive array-like input and return it as a 1-D array."""
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
        
        if np.any(array <= 0):
            raise ValueError(f"{name} must contain only positive values.")

        return array
    
    @classmethod
    def _to_scalar_array(
        cls,
        value: float,
        name: str
    ) -> np.ndarray:
        """Validate a finite scalar numeric input and return it as a NumPy scalar array."""
        try:
            array = np.asarray(value,dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be convertible to a numeric scalar.") from error

        if array.size == 0:
            raise ValueError(f"{name} cannot be empty.")
        
        if array.ndim != 0:
            raise ValueError(f"{name} must be scalar.")

        if not np.isfinite(array):
            raise ValueError(f"{name} must be finite.")

        return array
    
    @classmethod
    def _to_field_units(
        cls,
        array: np.ndarray
    ) -> np.ndarray:
        """Convert an SI array in meters to oilfield units in feet."""
        return array.copy() / cls.FEET_TO_METERS
    
    def _invalidate_spatial_cache(self) -> None:
        """Clear subclass caches derived from spatial arrays."""
        return None
    
    @property
    def radii(self):
        """Returns radii in feet."""
        return self._to_field_units(self._radii)

    @radii.setter
    def radii(self,value):
        """Sets radii after converting from feet to meters."""
        array = self._to_positive_array(value, "radii")
        
        if not np.all(np.diff(array) > 0):
            raise ValueError("Radii must be strictly increasing.")
        
        self._radii = array * self.FEET_TO_METERS
        self._invalidate_spatial_cache()

    @property
    def zdelta(self):
        """Returns z-direction grid cell size in feet."""
        return self._to_field_units(self._zdelta)

    @zdelta.setter
    def zdelta(self,value):
        """Sets z-direction grid cell size after converting from feet to meters."""
        self._zdelta = self._to_positive_array(value, "zdelta") * self.FEET_TO_METERS
        self._invalidate_spatial_cache()

    @property
    def tdelta(self):
        """Returns theta-direction grid cell angle in radians."""
        return self._tdelta.copy()

    @tdelta.setter
    def tdelta(self,value):
        """Sets theta-direction grid cell angle."""
        array = self._to_positive_array(value,"tdelta")

        if not np.isclose(np.sum(array), 2*np.pi):
            raise ValueError("Sum of tdelta values must equal 2*pi radians.")

        self._tdelta = array
        self._invalidate_spatial_cache()

    @property
    def depth(self):
        """Returns the reservoir-domain top-surface depth in feet."""
        return self._to_field_units(self._depth)

    @depth.setter
    def depth(self,value):
        """Sets reservoir-domain top-surface depth after converting feet to meters."""
        self._depth = self._to_scalar_array(value,"depth") * self.FEET_TO_METERS
        self._invalidate_spatial_cache()
