from prodpy.resprops import Layer, Fluid

from ._porous_media import PorousMedia

class BaseSolver(PorousMedia):
	"""
	Base class for analytical transient-flow solvers based on the diffusivity equation.

	This class combines porous-media geometry from :class:`PorousMedia` with
	reservoir rock and fluid properties required to compute derived quantities
	such as total compressibility, hydraulic diffusivity, and pore volume.

	The class is intended to be subclassed by specific analytical solutions,
	for example infinite-acting radial flow, pseudo-steady-state flow, linear
	flow, or other diffusivity-equation-based models.

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
	**kwargs
		Additional keyword arguments passed to the parent :class:`PorousMedia`
		class.

	Notes
	-----
	This base class assumes that the ``Layer`` and ``Fluid`` objects expose
	internal attributes such as ``_perm``, ``_poro``, ``_comp``, and ``_visc``.
	Subclasses should implement the ``__call__`` method to evaluate a specific
	analytical solution.

	"""
	FT_TO_METER = 0.3048
	PSI_TO_PA = 6894.757293168
	DAY_TO_SEC = 24*60*60

	def __init__(self, *args, layer:Layer=None, fluid:Fluid=None, **kwargs):
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

		self._layer = layer
		self._fluid = fluid

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

	@property
	def fluid(self):
		"""Fluid: Reservoir fluid-property object."""
		return self._fluid

	@property
	def tcomp(self):
		"""Total compressibility in 1/psi"""
		return self._tcomp*self.PSI_TO_PA

	@property
	def _tcomp(self):
		"""Total compressibility in 1/Pa"""
		return self.layer._comp+self.fluid._comp

	@property
	def hdiff(self):
		"""Hydraulic diffusivity in ft2/day."""
		return self._hdiff*self.DAY_TO_SEC/self.FT_TO_METER**2

	@property
	def _hdiff(self):
		"""Hydraulic diffusivity in m2/sec."""
		denominator = self.layer._poro*self.fluid._visc*self._tcomp

		return self.layer._perm/denominator[:, None]

	@property
	def vpore(self):
		"""Pore volume in ft3."""
		return self._vpore / (self.FT_TO_METER**3)

	@property
	def _vpore(self):
		"""Pore volume in m3."""
		return self._volume*self.layer._poro
	