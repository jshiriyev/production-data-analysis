import numpy as np

class Layer():
	"""
	Reservoir porous-media properties with oilfield-unit public access
	and SI-unit internal storage for the given pressure and temperature.

	Public properties use oilfield units:
	xperm, yperm, zperm, perm : mD
	comp : 1/psi
	press : psi

	Internal underscored properties use SI units:
	_xperm, _yperm, _zperm, _perm : m2
	_comp : 1/Pa
	_press : Pa

	Permeability, porosity, compressibility, and pressure may be scalar
	or array-like. Array-like values represent grid-cell properties.
	"""
	MD_TO_M2 = 9.869233e-16
	PSI_TO_PA = 6894.757293168

	def __init__(self, *args, poro=None, comp=None, press=None, allow_inf: bool = False, allow_nan: bool = True, **kwargs):
		"""
		Initializes a reservoir rock with the following petrophysical parameters:

		Parameters
		----------
		*args and **kwargs : Passed to self.set_permeability(*args,**kwargs)
		
		poro 	: float or np.ndarray of floats, optional
			Porosity of the rock, dimensionless

		comp 	: float or np.ndarray of floats, optional
			Isothermal compressibility factor of rock, 1/psi

		press   : float or np.ndarray of floats, optional
			Pressure at which properties are defined, psi

		allow_inf : bool, optional
			Whether to allow positive or negative infinity values in permeability,
			porosity, compressibility, or pressure. Default is False.
		
		allow_nan : bool, optional
			Whether to allow NaN values in permeability, porosity, compressibility,
			or pressure. Default is True.

		"""
		self.settings = {
			"allow_inf": allow_inf,
			"allow_nan": allow_nan,
		}

		self.set_permeability(*args, **kwargs)

		self.poro = poro
		self.comp = comp
		self.press = press

	def __repr__(self):
		"""Return a concise constructor-style representation of the layer."""

		if self.is_isotropic:
			args = [
				f"perm={self._repr_value(self.xperm)}",
				f"is_isotropic=True",
			]
		else:
			args = [
				f"xperm={self._repr_value(self.xperm)}",
				f"yperm={self._repr_value(self.yperm)}",
				f"zperm={self._repr_value(self.zperm)}",
			]

		args.extend([
			f"poro={self._repr_value(self.poro)}",
			f"comp={self._repr_value(self.comp)}",
			f"press={self._repr_value(self.press)}",
		])

		return f"{type(self).__name__}({', '.join(args)})"
	
	@staticmethod
	def _repr_value(value):
		"""Format scalar and array values for __repr__."""

		if value is None:
			return "None"

		arr = np.asarray(value)

		if arr.size == 1:
			scalar = float(arr.ravel()[0])

			if np.isnan(scalar):
				return "np.nan"
			if np.isposinf(scalar):
				return "np.inf"
			if np.isneginf(scalar):
				return "-np.inf"

			return repr(scalar)

		return (
			"np.array("
			+ np.array2string(
				arr,
				precision=6,
				separator=", ",
				threshold=8,
				edgeitems=2,
			)
			+ ")"
		)

	def set_permeability(self,xperm,*,yperm=None,zperm=None,yreduce:float=1.,zreduce:float=1.):
		"""
		Set directional permeability values.

		Parameters
		----------
		xperm 	: permeability in x-direction, mD.
		yperm   : permeability in y direction, mD. If omitted, computed as xperm * yreduce.
		zperm   : permeability in z direction, mD. If omitted, computed as xperm * zreduce.

		yreduce : yperm to xperm ratio, dimensionless
			Ratio used to compute yperm from xperm when yperm is omitted.

		zreduce : zperm to xperm ratio, dimensionless
			Ratio used to compute zperm from xperm when zperm is omitted.

		Raises
		------
		ValueError
			If values are negative, non-finite, empty, or incompatible in length.

		"""
		self._xperm = self._validate_range(xperm, 0, name="xperm", **self.settings)*self.MD_TO_M2

		size = self._xperm.size

		yreduce = self._validate_range(yreduce, 0, name="yreduce").item()
		zreduce = self._validate_range(zreduce, 0, name="zreduce").item()

		self._yperm = (
			self._xperm * yreduce
			if yperm is None
			else self._validate_range(yperm, 0, size=size, full=True, name="yperm", **self.settings)*self.MD_TO_M2
		)

		self._zperm = (
			self._xperm * zreduce
			if zperm is None
			else self._validate_range(zperm, 0, size=size, full=True, name="zperm", **self.settings)*self.MD_TO_M2
		)

	@classmethod
	def _validate_range(cls, value, *args,
		size:int|None = None,
		full:bool = False,
		name:str|None = None,
		lower_limit:float|None = None,
		upper_limit:float|None = None,
		include_lower_limit:bool = True,
		include_upper_limit:bool = True,
		allow_inf: bool = False,
		allow_nan: bool = True,
		):
		arr = cls._as_float_array(value, size, full, name)

		if not allow_inf and np.any(np.isinf(arr)):
			raise ValueError(f"{name} must not contain positive or negative infinity.")
		
		if not allow_nan and np.any(np.isnan(arr)):
			raise ValueError(f"{name} must not contain NaN values.")

		if len(args) > 2:
			raise ValueError("Only lower_limit and upper_limit can be passed positionally.")

		if len(args) == 1:
			lower_limit = args[0]

		if len(args) == 2:
			lower_limit, upper_limit = args

		if lower_limit is not None:
			invalid = arr < lower_limit if include_lower_limit else	arr <= lower_limit
			message = "smaller than" if include_lower_limit else "smaller than or equal to"

			if np.any(invalid):
				raise ValueError(f"{name} cannot be {message} {lower_limit}.")
		
		if upper_limit is not None:
			invalid = arr > upper_limit if include_upper_limit else arr >= upper_limit
			message = "larger than" if include_upper_limit else "larger than or equal to"

			if np.any(invalid):
				raise ValueError(f"{name} cannot be {message} {upper_limit}.")
		
		return arr
	
	@staticmethod
	def _as_float_array(value, size:int|None = None, full:bool = False, name:str|None = None):
		"""Convert input to a 1D NumPy float array and optionally validate its size."""
		try:
			arr = np.asarray(value, dtype=float).ravel()
		except (TypeError, ValueError) as exc:
			raise TypeError(f"{name} must be convertible to a float array.") from exc

		if arr.size == 0:
			raise ValueError(f"{name} cannot be empty.")
		
		if size is not None:
			if size <= 0:
				raise ValueError("size must be a positive integer.")

			if arr.size == 1:
				if full:
					arr = np.full(size, arr.item(), dtype=float)

			elif arr.size != size:
				raise ValueError(
					f"{name} must be scalar or have length {size}; got length {arr.size}."
				)

		return arr

	@property
	def perm(self):
		"""Permeability matrix in mD, with columns [xperm, yperm, zperm]."""
		return self._perm/self.MD_TO_M2

	@property
	def _perm(self):
		"""Permeability matrix in m2, with columns [xperm, yperm, zperm]."""
		return np.column_stack((self._xperm,self._yperm,self._zperm))

	@property
	def xperm(self):
		"""Getter for the reservoir permeability in x-direction."""
		return None if self._xperm is None else self._xperm/self.MD_TO_M2

	@property
	def yperm(self):
		"""Getter for the reservoir permeability in y-direction."""
		return self._yperm/self.MD_TO_M2

	@property
	def zperm(self):
		"""Getter for the reservoir permeability in z-direction."""
		return self._zperm/self.MD_TO_M2
	
	@property
	def is_isotropic(self) -> bool:
		"""
		Return True if the layer permeability is isotropic.

		A layer is considered isotropic when the directional permeabilities are
		equal cell-by-cell:

			xperm == yperm == zperm

		If all three directional permeabilities are NaN for the same cell,
		that cell is treated as consistently undefined and does not make the
		layer anisotropic.

		Returns
		-------
		bool
			True if x-, y-, and z-direction permeabilities are equal for every
			cell, including matching NaN triplets; otherwise False.
		"""
		return bool(
			np.allclose(self._xperm, self._yperm, rtol=1e-12, atol=0.0, equal_nan=True)
			and np.allclose(self._xperm, self._zperm, rtol=1e-12, atol=0.0, equal_nan=True)
		)

	@property
	def size(self):
		return self._xperm.size

	@property
	def poro(self):
		"""Getter for the porosity values."""
		return self._poro

	@poro.setter
	def poro(self,value):
		"""Setter for the porosity values if value is available; otherwise sets None."""
		self._poro = None if value is None else self._validate_range(value, 0, 1, size=self.size, name="poro", **self.settings)

	@property
	def comp(self):
		"""Getter for the compressibility value in 1/psi if available; otherwise, returns None."""
		return None if self._comp is None else self._comp*self.PSI_TO_PA

	@comp.setter
	def comp(self,value):
		"""Setter for the compressibility value if value (in 1/psi) is available; otherwise sets None."""
		self._comp = None if value is None else (
			self._validate_range(value, 0, size=self.size, name="comp", **self.settings)/self.PSI_TO_PA
		)

	@property
	def press(self):
		"""Getter for the pressure value in psi if available; otherwise, returns None."""
		return None if self._press is None else self._press/self.PSI_TO_PA

	@press.setter
	def press(self,value):
		"""Setter for the pressure value if value (in psi) is available; otherwise sets None."""
		self._press = None if value is None else (
			self._validate_range(value, 0, size=self.size, name="press", **self.settings)*self.PSI_TO_PA
		)

if __name__ == "__main__":

	rrock = Layer((10,15,20),poro=(0.1,0.2,0.3),yreduce=0.5,zreduce=0.1)

	print(rrock.xperm)
	print(rrock.yperm)
	print(rrock.zperm)

	print(rrock.perm)
	
	print(rrock.poro)

