import logging

from respy import Layer, Fluid

class SolverObj():
	"""Base solver of the diffusivity equation."""

	def __init__(self,*,layer:Layer=None,fluid:Fluid=None,tcomp:float=None):
		"""Initialization with layer and fluid information, and optionally,
		with total compressibility.

		"""
		self.layer = layer
		self.fluid = fluid
		self.tcomp = tcomp

	@property
	def layer(self):
		"""Getter for the reservoir rock properties."""
		return self._layer

	@layer.setter
	def layer(self,value):
		"""Setter for the reservoir rock properties."""
		self._layer = value

	@property
	def fluid(self):
		"""Getter for the reservoir fluid properties."""
		return self._fluid

	@fluid.setter
	def fluid(self,value):
		"""Setter for the reservoir fluid properties."""
		self._fluid = value

	@property
	def tcomp(self):
		"""Getter for the total compressibility."""
		return None if self._tcomp is None else self._tcomp*6894.76

	@tcomp.setter
	def tcomp(self,value):
		"""Setter for the total compressibility."""
		if value is None:
			try:
				self._tcomp = self.layer._comp+self.fluid._comp
			except Exception as e:
				logging.warning(f"Missing attribute when calculating total compressibility: {e}")
		else:
			self._tcomp = value/6894.76

	@property
	def hdiff(self):
		"""Getter for the hydraulic diffusivity."""
		if not hasattr(self,"_hdiff"):
			self.hdiff = None

		return self._hdiff*(3.28084**2)*(24*60*60)

	@hdiff.setter
	def hdiff(self,value):
		"""Setter for the hydraulic diffusivity."""
		self._hdiff = (self.layer._perm)/(self.layer._poro*self.fluid._visc*self._tcomp)
	