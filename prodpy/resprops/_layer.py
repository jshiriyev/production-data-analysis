import numpy as np

class Layer():
	"""
	Base class that defines constant reservoir rock properties
	at the given pressure and temperature.
	
	"""

	def __init__(self,*args,poro=None,comp=None,press=None,**kwargs):
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

		"""
		self.set_permeability(*args,**kwargs)

		self.poro  = poro
		self.comp  = comp
		self.press = press

	def set_permeability(self,xperm,*,yperm=None,zperm=None,yreduce:float=1.,zreduce:float=1.):
		"""Assigns the permeability values in mD to the grids.

		xperm 	: permeability in x-direction, mD
		yperm   : permeability in y direction, mD
		zperm   : permeability in z direction, mD

		yreduce : yperm to xperm ratio, dimensionless
		zreduce : zperm to xperm ratio, dimensionless

		"""
		self.xperm = xperm
		self.yperm = self.xperm*yreduce if yperm is None else np.ravel(yperm).astype(float)
		self.zperm = self.xperm*zreduce if zperm is None else np.ravel(zperm).astype(float)

	@property
	def perm(self):
		"""Getter for the reservoir permeability."""
		if not hasattr(self,"_perm"):
			self.perm = None

		return self._perm/9.869233e-16

	@perm.setter
	def perm(self,value):
		"""Setter for the reservoir permeability."""
		self._perm = np.column_stack((self._xperm,self._yperm,self._zperm))

	@property
	def xperm(self):
		"""Getter for the reservoir permeability in x-direction."""
		return None if self._xperm is None else self._xperm/9.869233e-16

	@xperm.setter
	def xperm(self,value):
		"""Setter for the reservoir permeability in x-direction."""
		self._xperm = np.ravel(value).astype(float)*9.869233e-16

	@property
	def yperm(self):
		"""Getter for the reservoir permeability in y-direction."""
		return self._yperm/9.869233e-16

	@yperm.setter
	def yperm(self,value):
		"""Setter for the reservoir permeability in y-direction."""
		self._yperm = value*9.869233e-16

	@property
	def zperm(self):
		"""Getter for the reservoir permeability in z-direction."""
		return self._zperm/9.869233e-16

	@zperm.setter
	def zperm(self,value):
		"""Setter for the reservoir permeability in z-direction."""
		self._zperm = value*9.869233e-16
	
	@property
	def poro(self):
		"""Getter for the porosity values."""
		return self._poro

	@poro.setter
	def poro(self,value):
		"""Setter for the porosity values if value is available; otherwise sets None."""
		self._poro = None if value is None else np.ravel(value).astype(float)

	@property
	def comp(self):
		"""Getter for the compressibility value in 1/psi if available; otherwise, returns None."""
		return None if self._comp is None else self._comp*6894.75729

	@comp.setter
	def comp(self,value):
		"""Setter for the compressibility value in 1/Pa if value is available; otherwise sets None."""
		self._comp = None if value is None else np.ravel(value).astype(float)/6894.75729

	@property
	def press(self):
		"""Getter for the pressure value in psi if available; otherwise, returns None."""
		return None if self._press is None else self._press/6894.76

	@press.setter
	def press(self,value):
		"""Setter for the pressure value in Pa if value is available; otherwise sets None."""
		self._press = None if value is None else np.ravel(value).astype(float)*6894.76

if __name__ == "__main__":

	rrock = Layer((10,15,20),poro=(0.1,0.2,0.3),yreduce=0.5,zreduce=0.1)

	print(rrock.xperm)
	print(rrock.yperm)
	print(rrock.zperm)

	print(rrock.perm)
	
	print(rrock.poro)

