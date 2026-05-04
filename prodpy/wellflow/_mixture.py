from respy import Fluid

from .pressure_drop._pipe import Pipe
from .pressure_drop._darcy_weisbach import DarcyWeisbach

class Mixture():

	def __init__(self,pipe:Pipe,gas:Fluid,liq:Fluid):
		"""Initializing gas-liquid mixture model."""
		self.pipe = pipe
		
		self.gas = gas
		self.liq = liq

	def get(self,grate,lrate,slip:float=1.,**kwargs):
		"""Calculates the head loss due to friction using the selected model."""
		fluid = self.fluid(grate,lrate,slip)
		model = self.model(fluid)

		return model.get(fluid._rate,**kwargs)

	def fluid(self,grate,lrate,slip:float=1.):
		"""Returns mixture fluid with density, viscosity, quality, voidage, and flow rate properties."""
		gmass = grate*self.gas._rho
		lmass = lrate*self.liq._rho

		x = self.quality(gmass,lmass)
		mu = self.visc(self.gas.visc,self.liq.visc,x)

		a = self.voidage(grate,lrate,slip=slip)
		rho = self.rho(self.gas.rho,self.liq.rho,a)

		fluid = Fluid(mu,rho=rho)

		fluid.quality = x
		fluid.voidage = a

		fluid._rate = (gmass+lmass)/(fluid._rho)

		return fluid

	def model(self,fluid):
		"""Returns pressure drop model for the given fluid."""
		return DarcyWeisbach(self.pipe,fluid)

	@staticmethod
	def quality(massG,massL):
		"""Returns mass quality of gas."""
		return massG/(massG+massL)

	@staticmethod
	def voidage(voidG,voidL,slip:float=1.):
		"""Returns volume quality of gas."""
		return voidG/(voidG+slip*voidL)
	
	@staticmethod
	def rho(rhoG,rhoL,voidage:float=0.5):
		"""Returns mixture density based on void fraction."""
		return rhoG*voidage+rhoL*(1-voidage)
	   
	@staticmethod
	def visc(viscG,viscL,x:float=0.5):
		"""Returns mixture viscosity based on mass quality."""
		return viscG*x+viscL*(1-x)