import numpy

from scipy import optimize

class DranchukPurvisRobinson():
	"""
	Dranchuk-Purvis-Robinson Method: Provides function to calculate
	gas compressibility fractor (z-factor);

	Recommended for applications where:
	0.2 < Pr < 3.0 and 1.05 < Tr < 3.0.
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

		self.T1 = self.get_T1(self.treduced)
		self.T2 = self.get_T2(self.treduced)
		self.T3 = self.get_T3(self.treduced)
		self.T4 = self.get_T4(self.treduced)

		self.A8 = 0.68446549

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
	def get_T1(treduced):
		A1 =  0.31506237
		A2 = -1.0467099
		A3 = -0.57832720
		return A1+A2/treduced+A3/treduced**3

	@staticmethod
	def get_T2(treduced):
		A4 =  0.53530771
		A5 = -0.61232032
		return A4+A5/treduced

	@staticmethod
	def get_T3(treduced):
		A5 = -0.61232032
		A6 = -0.10488813
		return A5*A6/treduced

	@staticmethod
	def get_T4(treduced):
		A7 = 0.68157001
		return A7/treduced**3

	def __call__(self,press:numpy.ndarray,derivative:bool=False):

		preduced = self.preduced(press)

		if self.treduced<1.05 or self.treduced>3 or numpy.any(preduced<0.2) or numpy.any(preduced>3.0):
			raise Warning("Dranchuk-Purvis-Robinson Method is not recommended")

		T5 = (0.27*preduced)/self.treduced

		residual = lambda rhor: self.zvalue(rhor)-T5/rhor
		resprime = lambda rhor: self.zprime(rhor)+T5/rhor**2

		rhor = optimize.newton(residual,T5,fprime=resprime)

		if derivative:
			return T5/rhor,self.zprime(rhor)

		return T5/rhor

	def zvalue(self,rhor):
		"""Internal function to calculate z factor when the class is called."""
		return 1+self.T1*rhor+self.T2*rhor**2+self.T3*rhor**5\
				+self.T4*rhor**2*(1+self.A8*rhor**2)*numpy.exp(-self.A8*rhor**2)

	def zprime(self,rhor):
		"""Internal function to calculate z prime when the class is called."""
		return self.T1+2*self.T2*rhor+5*self.T3*rhor**4\
			+2*self.T4*rhor*(1+self.A8*rhor**2-self.A8**2*rhor**4)*numpy.exp(-self.A8*rhor**2)


	