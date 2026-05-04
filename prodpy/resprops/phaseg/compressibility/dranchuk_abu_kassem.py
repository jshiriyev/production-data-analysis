import numpy

from scipy import optimize

class DranchukAbuKassem():
	"""
	Dranchuk-Abu-Kassem Method represents the Standing-Katz correlation:
	
	0.2 < Pr < 15 and 0.7 < Tr < 3.0 within 1 % error
	 15 < Pr < 30 within 3% error.

	It provides function to calculate gas compressibility fractor (z-factor);
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

		self.R1 = self.get_R1(self.treduced)
		self.R2 = self.get_R2(self.treduced)
		self.R3 = self.get_R3(self.treduced)
		self.R4 = self.get_R4(self.treduced)

		self.A11 = 0.7210

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
	def get_R1(treduced):
		A1,A2,A3,A4,A5 = 0.3265,-1.0700,-0.5339,0.01569,-0.05165
		return A1+A2/treduced+A3/treduced**3+A4/treduced**4+A5/treduced**5

	@staticmethod
	def get_R2(treduced):
		A6,A7,A8 = 0.5475,-0.7361,0.1844
		return A6+A7/treduced+A8/treduced**2

	@staticmethod
	def get_R3(treduced):
		A7,A8,A9 = -0.7361,0.1844,0.1056
		return A9*(A7/treduced+A8/treduced**2)

	@staticmethod
	def get_R4(treduced):
		A10 = 0.6134
		return A10/treduced**3

	def __call__(self,press:numpy.ndarray,derivative:bool=False):

		preduced = self.preduced(press)

		if numpy.any(preduced>30):
			raise Warning("Dranchuk-Abu-Kassem method is not recommended.")
		elif numpy.any(preduced<0.2):
			raise Warning("Dranchuk-Abu-Kassem method is not recommended.")
		elif numpy.any(preduced<15) and (self.treduced<0.7 or self.treduced>3.0):
			raise Warning("Dranchuk-Abu-Kassem method is not recommended.")
			
		R5 = (0.27*preduced)/self.treduced

		residual = lambda rhor: self.zvalue(rhor)-R5/rhor
		resprime = lambda rhor: self.zprime(rhor)+R5/rhor**2

		rhor = optimize.newton(residual,R5,fprime=resprime)

		if derivative:
			return R5/rhor,self.zprime(rhor)

		return R5/rhor

	def zvalue(self,rhor):
		"""Internal function to calculate z factor when the class is called."""
		return 1+self.R1*rhor+self.R2*rhor**2-self.R3*rhor**5\
				+self.R4*(1+self.A11*rhor**2)*rhor**2*numpy.exp(-self.A11*rhor**2)

	def zprime(self,rhor):
		"""Internal function to calculate z prime when the class is called."""
		return self.R1+2*self.R2*rhor-5*self.R3*rhor**4\
			+2*self.R4*rhor*(1+self.A11*rhor**2*(1-self.A11*rhor**2))*numpy.exp(-self.A11*rhor**2)

	