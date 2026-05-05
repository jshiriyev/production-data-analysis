import numpy as np

from scipy import optimize

from ._pressure_drop import PressureDrop

class DarcyWeisbach(PressureDrop):
	"""
	Computes head loss and friction factor using the Darcy-Weisbach equation.

	Inherits from:
		PressureDrop: A base class handling pipe and fluid properties.

	Methods:
		get(flow_rate: float) -> float:
			Calculates the head loss due to friction.
		friction(flow_rate: float, method="colebrook") -> float:
			Computes the Darcy friction factor based on flow regime.
		reynolds(flow_rate: float) -> float:
			Computes the Reynolds number.
		
	Static Methods:
		colebrook(Re: float, epd: float) -> float:
			Computes friction factor using the Colebrook equation.
		haaland(Re: float, epd: float) -> float:
			Computes friction factor using the Haaland equation.
		chen(Re: float, epd: float) -> float:
			Computes friction factor using the Chen equation.
	"""
	LOWER_REYNOLDS_LIMIT = 2000
	UPPER_REYNOLDS_LIMIT = 4000

	def __init__(self,*args,**kwargs):
		"""
		Initializes the DarcyWeisbach class by inheriting from PressureDrop.
		
		Args:
			*args: Positional arguments passed to the parent class.
			**kwargs: Keyword arguments passed to the parent class.

		"""
		super().__init__(*args,**kwargs)

	def get(self,flow_rate:float|np.ndarray,**kwargs) -> float:
		"""Calculates the head loss due to friction using the Darcy-Weisbach equation."""
		fD = self.friction(flow_rate,**kwargs)

		v = flow_rate/self.pipe.csa
		g = 9.80665 # Gravitational acceleration (m/s²)

		return (fD * self.pipe.length * v**2) / (2*g*self.pipe.diam)

	def friction(self,flow_rate:float|np.ndarray,method:str="colebrook",**kwargs):
		"""Computes the Darcy-Weisbach friction factor based on the flow regime.
		
		Args:
			flow_rate (float): Volumetric flow rate in cubic meters per second (m³/s).
			method (str, optional): Friction factor correlation method: "colebrook", "haaland", or "chen" (default="colebrook").

		Returns:
			float: Darcy friction factor.

		Raises:
			ValueError: If the Reynolds number falls in the transition regime.

		"""
		Re = self.reynolds(flow_rate)

		fD = np.empty_like(Re)

		c1 = Re<self.LOWER_REYNOLDS_LIMIT
		c2 = np.logical_and(Re>=self.LOWER_REYNOLDS_LIMIT,Re<=self.UPPER_REYNOLDS_LIMIT)
		c3 = Re>self.UPPER_REYNOLDS_LIMIT

		fD[c1] = 64/Re[c1]
		fD[c2] = np.nan
		fD[c3] = getattr(self,method)(Re[c3],self.pipe.epd,**kwargs)

		return fD

	def reynolds(self,flow_rate:float|np.ndarray) -> float:
		"""Computes the Reynolds number for the given flow rate."""
		Q = np.ravel(flow_rate)

		self.__reynolds_number = (4*self.fluid._rho*Q)/(np.pi*self.fluid._visc*self.pipe.diam)

		return self.__reynolds_number

	@property
	def laminar(self):
		"""Returns True if the last calculated flow regime is laminar."""
		flag = np.zeros(self.__reynolds_number.shape,dtype=bool)
		flag[self.__reynolds_number<self.LOWER_REYNOLDS_LIMIT] = True
		
		return flag

	@property
	def turbulent(self):
		"""Returns True if the last calculated flow regime is turbulent."""
		flag = np.zeros(self.__reynolds_number.shape,dtype=bool)
		flag[self.__reynolds_number>self.UPPER_REYNOLDS_LIMIT] = True
		
		return flag

	@staticmethod
	def blasius(Re:float|np.ndarray,*args,**kwargs) -> float:
		"""Computes the Darcy-Weisbach friction factor for turbulent flow in smooth pipes using Blasius correlation."""
		return 0.3164/Re**0.25

	@staticmethod
	def colebrook(Re:float|np.ndarray,epd:float,**kwargs) -> float:
		"""Computes the Darcy-Weisbach friction factor using the Colebrook equation."""
		inner = lambda phi,Re,epd: epd/3.7+2.51/Re/np.sqrt(phi)
		func  = lambda phi,Re,epd: 1/np.sqrt(phi)+2*np.log10(inner(phi,Re,epd))
		prime = lambda phi,Re,epd: -1/(2*phi**(3/2))*(1+2.18/(Re*inner(phi,Re,epd)))
		
		return optimize.newton(func,64/Re,prime,args=(Re,epd),**kwargs)

	@staticmethod
	def haaland(Re:float|np.ndarray,epd:float) -> float:
		"""Computes the Darcy-Weisbach friction factor using the Haaland equation."""
		return 1/(-1.8*np.log10((epd/3.7)**1.11+6.9/Re))**2

	@staticmethod
	def chen(Re:float|np.ndarray,epd:float) -> float:
		"""Computes the Darcy-Weisbach friction factor using the Chen equation."""
		return 1/(-2*np.log10(epd/3.7065-5.0452/Re*np.log10((epd**1.1098)/2.8257+5.8506/Re**0.8981)))**2