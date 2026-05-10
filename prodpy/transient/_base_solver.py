import numpy as np

from prodpy.resprops import Layer, Fluid
from prodpy.porsim.prerun import Borehole

from ._porous_media import PorousMedia

class BaseSolver(PorousMedia):
	"""
	Shared base for analytical single-well diffusivity-equation solvers.

	BaseSolver combines drainage geometry, rock properties, fluid properties,
    borehole geometry, and initial pressure. Public properties use oilfield
    units; underscored properties use SI units for equation evaluation.

	Parameters
	----------
	*args
		Positional arguments passed to the parent :class:`PorousMedia` class.
	layer : Layer, optional
		Reservoir rock/property object. Expected to provide properties such as
		permeability, porosity, and rock compressibility through the internal
		attributes used by this class.
	fluid : Fluid, optional
		Reservoir fluid-property object. Expected to provide viscosity and
		fluid compressibility through the internal attributes used by this class.
	borehole : Borehole
        Wellbore geometry and operating constraints.
    pinit : float
        Initial reservoir pressure, psi.
	**kwargs
		Additional keyword arguments passed to the parent :class:`PorousMedia`
		class.

	Notes
	-----
	Public properties use oilfield units; underscored properties use SI units.

	This class assumes single-phase, slightly compressible flow and scalar
    effective properties and isotropic permeability. Subclasses implement a
	specific flow regime by overriding `__call__` or `solve`.

	BaseSolver represents a scheduled oil-rate-controlled borehole. All borehole
	constraints must be `orate` constraints.

	"""
	FT_TO_METER = 0.3048
	PSI_TO_PA = 6894.757293168
	DAY_TO_SEC = 24*60*60
	_GRAVITY = 9.80665

	def __init__(self,
		*args,
		layer:Layer | None = None,
		fluid:Fluid | None = None,
		borehole:Borehole | None = None,
		pinit:float | None = None,
		**kwargs
	) -> None:
		"""Initialize the base solver with porous-media, layer, and fluid properties.

		Parameters
		----------
		*args
			Positional arguments passed to :class:`PorousMedia`.
		layer : Layer, optional
			Reservoir layer object containing rock and petrophysical properties.
		fluid : Fluid, optional
			Fluid object containing viscosity and compressibility properties.
		**kwargs
			Additional keyword arguments passed to :class:`PorousMedia`.

		"""
		super().__init__(*args,**kwargs)

		self.layer = layer
		self.fluid = fluid
		self.borehole = borehole

		self.pinit = pinit

	def __call__(self, tcomp: float = None):
		"""
		Evaluate the analytical solution.

		This method should be implemented by subclasses. The base class only
		defines the common reservoir and fluid-property calculations required
		by analytical diffusivity-equation solvers.

		Parameters
		----------
		tcomp : float, optional
			Optional total compressibility override.

		Raises
		------
		NotImplementedError
			Subclasses should override this method.
		"""
		raise NotImplementedError(
			"__call__ must be implemented by subclasses of BaseSolver."
		)

	@property
	def layer(self):
		"""Layer: Reservoir rock and layer-property object."""
		return self._layer
	
	@layer.setter
	def layer(self, value: Layer | None) -> None:
		"""
		Set the reservoir layer object.

		The layer may be temporarily None to allow delayed assignment, but
		calculations requiring layer properties will raise a clear error until
		a valid Layer object is assigned.
		"""
		if value is None:
			self._layer = None
			return

		if not isinstance(value, Layer):
			raise TypeError(
				f"layer must be a Layer instance or None; got {type(value).__name__}."
			)

		if not value.is_isotropic:
			raise ValueError(
				"layer must be isotropic for this solver: xperm, yperm, and zperm "
				"must be equal."
			)

		if value.poro is None:
			raise ValueError("layer.poro must be set before using this solver.")

		self._layer = value
	
	@property
	def perm(self):
		"""Permeability in mD."""
		layer = self._require_layer()

		if not layer.is_isotropic:
			raise ValueError(
				"This solver requires isotropic permeability, but layer is anisotropic."
			)

		return layer.perm[:, 0]
	
	@property
	def _perm(self):
		"""Permeability in m2."""
		layer = self._require_layer()

		if not layer.is_isotropic:
			raise ValueError(
				"This solver requires isotropic permeability, but layer is anisotropic."
			)

		return layer._perm[:,0]

	@property
	def fluid(self):
		"""Fluid: Reservoir fluid-property object."""
		return self._fluid
	
	@fluid.setter
	def fluid(self, value: Fluid | None) -> None:
		"""
		Set the reservoir fluid object.

		The fluid may be temporarily None to allow delayed assignment, but
		calculations requiring fluid properties will raise a clear error until
		a valid Fluid object is assigned.
		"""
		if value is None:
			self._fluid = None
			return

		if not isinstance(value, Fluid):
			raise TypeError(
				f"fluid must be a Fluid instance or None; got {type(value).__name__}."
			)

		required_attrs = ("_visc", "_mobil")

		missing = [
			attr for attr in required_attrs
			if not hasattr(value, attr) or getattr(value, attr) is None
		]

		if missing:
			raise ValueError(
				"fluid is missing required properties for this solver: "
				+ ", ".join(missing)
			)

		self._fluid = value
	
	@property
	def borehole(self):
		"""Borehole: Well object containing radius, skin, and constraints."""
		return self._borehole
	
	@borehole.setter
	def borehole(self, value: Borehole | None) -> None:
		"""
		Set the borehole object.

		The borehole may be temporarily None to allow delayed assignment, but
		calculations requiring borehole geometry or constraints will raise a
		clear error until a valid Borehole object is assigned.
		"""
		if value is None:
			self._borehole = None
			return

		if not isinstance(value, Borehole):
			raise TypeError(
				f"borehole must be a Borehole instance or None; got {type(value).__name__}."
			)

		required_attrs = ("_radius", "_area", "constraints")

		missing = [
			attr for attr in required_attrs
			if not hasattr(value, attr) or getattr(value, attr) is None
		]

		if missing:
			raise ValueError(
				"borehole is missing required properties for this solver: "
				+ ", ".join(missing)
			)
		
		constraints = value.constraints

		if not constraints:
			raise ValueError(
				"borehole must contain at least one oil-rate constraint with mode='orate'."
			)

		if not all(c.mode == "orate" for c in constraints):
			raise ValueError(
				"BaseSolver only supports oil-rate-controlled boreholes; "
				"all constraints must have mode='orate'."
			)

		self._borehole = value
	
	@property
	def rate(self):
		"""Oil-rate schedule limits in STB/day."""
		borehole = self._require_borehole()
		return np.array([c.limit for c in borehole.constraints])
	
	@property
	def _rate(self):
		"""Oil-rate schedule limits in m3/s."""
		borehole = self._require_borehole()
		return np.array([c._limit for c in borehole.constraints])
	
	@property
	def pinit(self):
		"""Initial pressure in psi."""
		return None if self._pinit is None else self._pinit/self.PSI_TO_PA
	
	@pinit.setter
	def pinit(self, value):
		"""Initial pressure in psi."""
		self._pinit = None if value is None else Layer._validate_range(
			value, 0, name="pinit", include_lower_limit = False,
			).item()*self.PSI_TO_PA

	@property
	def tcomp(self):
		"""Total compressibility in 1/psi"""
		return self._tcomp*self.PSI_TO_PA

	@property
	def _tcomp(self):
		"""Total compressibility in 1/Pa"""
		layer, fluid, _ = self._require_ready()
		return layer._comp + fluid._comp

	@property
	def hdiff(self):
		"""Hydraulic diffusivity in ft2/day."""
		return self._hdiff*self.DAY_TO_SEC/self.FT_TO_METER**2

	@property
	def _hdiff(self):
		"""Hydraulic diffusivity in m2/sec."""
		layer, fluid, _ = self._require_ready()
		return self._perm / (layer._poro * fluid._visc * self._tcomp)

	@property
	def vpore(self):
		"""Pore volume in ft3."""
		return self._vpore / (self.FT_TO_METER**3)

	@property
	def _vpore(self):
		"""Pore volume in m3."""
		layer = self._require_layer()
		return self._volume * layer._poro
	
	@property
	def storage(self):
		"""Storage coefficient in ft3/psi."""
		return self._storage*self.PSI_TO_PA*self.FT_TO_METER**3

	@property
	def _storage(self):
		"""Storage coefficient in m3/Pa."""
		_, fluid, borehole = self._require_ready()
		return (borehole._area)/(fluid._rho * self._GRAVITY)
	
	@property
	def pcons(self):
		"""Constant for dimensionless pressure scaling in psi."""
		return self._pcons / self.PSI_TO_PA

	@property
	def _pcons(self):
		"""Constant for dimensionless pressure scaling in Pa."""
		fluid = self._require_fluid()
		return (self._rate)/(2*np.pi*self._perm*self._height*fluid._mobil)

	def tD(self, time: float):
		"""Dimensionless time for the diffusivity equation."""
		_, _, borehole = self._require_ready()

		try:
			time = float(time)
		except ValueError:
			raise ValueError("Time must be a numeric value.")
		
		time *= self.DAY_TO_SEC
		
		return (self._hdiff * time)/(borehole._radius**2)
	
	def CD(self):
		"""Dimensionless wellbore storage coefficient."""
		layer, _, borehole = self._require_ready()

		length = layer.poro*self._height
		volume = length*borehole._area

		return (self._storage)/(volume*self._tcomp)

	def pD(self, press: float):
		"""Dimensionless pressure."""
		if self._pinit is None:
			raise ValueError("Initial pressure (pinit) must be set to compute dimensionless pressure.")
		
		try:
			press = float(press)
		except ValueError:
			raise ValueError("Pressure must be a numeric value.")
		
		press *= self.PSI_TO_PA
		
		return (self._pinit-press)/self._pcons
	
	def _require_layer(self) -> Layer:
		"""Return layer if available; otherwise raise a clear error."""
		if self.layer is None:
			raise ValueError("layer must be set before using this solver.")
		return self.layer


	def _require_fluid(self) -> Fluid:
		"""Return fluid if available; otherwise raise a clear error."""
		if self.fluid is None:
			raise ValueError("fluid must be set before using this solver.")
		return self.fluid


	def _require_borehole(self) -> Borehole:
		"""Return borehole if available; otherwise raise a clear error."""
		if self.borehole is None:
			raise ValueError("borehole must be set before using this solver.")
		return self.borehole


	def _require_ready(self) -> tuple[Layer, Fluid, Borehole]:
		"""Return layer, fluid, and borehole after validating solver readiness."""
		return self._require_layer(), self._require_fluid(), self._require_borehole()
	


	