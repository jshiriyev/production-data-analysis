import numpy

from scipy import optimize

class HallYarborough():
	"""
	Hall Yarborough Method: Provides function to calculate
	gas compressibility fractor (z-factor);

	Not recommended	for pseudo-reduced temperatures less than one.
	"""

	def __init__(self,crit,temp):
		"""Z factor class that can be used for ResPy, returning compressibility
			factors for pressures when called. Initialization parameters are:

		crit    : tuple of (pcrit in psi, tcrit in Rankine)

		temp    : temperature value (Rankine) which will be used to calculate
				  fluid properties when the class is called.

		- Initialize the class with critical pressure and temperature,
		- Provide temperature at which properties will be calculated during the iterations,
		assuming isothermal conditions
		- Call the class to calculate z-factor and its prime at different pressure values.

		"""

		pcrit,tcrit = crit

		self.pcrit = pcrit
		self.tcrit = tcrit

		self.temp = temp

		if self.treduced<1:
			raise Warning("Hall Yarborough method is not recommended for Tr less than one.")

		self.X2 = self.get_X2(self.treduced)
		self.X3 = self.get_X3(self.treduced)
		self.X4 = self.get_X4(self.treduced)

	@property
	def pcrit(self):
		"""Critical Pressure in psi, underscore parameter is in SI units."""
		return self._pcrit/6894.76

	@pcrit.setter
	def pcrit(self,value:float):
		"""Critical Pressure in SI units, Pascal."""
		self._pcrit = value*6894.76

	@property
	def tcrit(self):
		"""Critical Temperature in Rankine, underscore parameter is in SI units."""
		return self._tcrit*(9./5)

	@tcrit.setter
	def tcrit(self,value:float):
		"""Critical Temperature in SI units, Kelvin."""
		self._tcrit = value*(5./9)

	@property
	def temp(self):
		"""Temperature in Rankine, underscore parameter is in SI units."""
		return self._temp*(9./5)

	@temp.setter
	def temp(self,value:float):
		"""Temperature in SI units, Kelvin."""
		self._temp = value*(5./9)

	@property
	def treduced(self):
		"""Returns reduced temperature (class property)."""
		return self.temp/self.tcrit

	def preduced(self,press:numpy.ndarray):
		"""Returns reduced pressure values for input pressure values in psi."""
		return numpy.asarray(press)/self.pcrit

	@staticmethod
	def get_X2(treduced):
		return 14.76/treduced-9.76/treduced**2+4.58/treduced**3

	@staticmethod
	def get_X3(treduced):
		return 90.7/treduced-242.2/treduced**2+42.4/treduced**3

	@staticmethod
	def get_X4(treduced):
		return 2.18+2.82/treduced

	def __call__(self,press:numpy.ndarray,derivative:bool=False):

		preduced = self.preduced(press)

		X1 = 0.06125*preduced/self.treduced*numpy.exp(-1.2*(1-1/self.treduced)**2)

		residual = lambda Y: self.zfunc(Y)*Y-X1
		resprime = lambda Y: self.zfunc(Y)+self.zprime(Y)*Y

		Y0 = 0.0125*preduced/self.treduced*numpy.exp(-1.2*(1-1/self.treduced)**2) # initial guess for Y

		Y = optimize.newton(residual,Y0,fprime=resprime)

		if derivative:
			return X1/Y,self.zprime(Y)

		return X1/Y

	def zfunc(self,Y):
		"""Internal function to calculate z factor when the class is called."""
		return (1+Y+Y**2+Y**3)/(1-Y)**3-self.X2*Y+self.X3*Y**(self.X4-1)

	def zprime(self,Y):
		"""Internal function to calculate z prime when the class is called."""
		return 4*(1+Y+Y**2)/(1-Y)**4-self.X2+self.X3*(self.X4-1)*Y**(self.X4-2)