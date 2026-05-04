import numpy

from scipy.optimize import minimize

class Tank():
	"""
	Material Balance tank model class.
	Use __call__ to create a new state of the model with pressure and production inputs.

	"""

	def __init__(self, Pi, N, G, Boi, Bgi, Rsi, cw, cf, Swi):

		self.Pi = Pi	# Initial pressure, psi
		self.N = N		# Initial (original) oil in place, STB
		self.G = G		# Initial gas-cap gas, scf
		self.Boi = Boi	# Initial oil formation volume factor, bbl/STB
		self.Bgi = Bgi	# Initial gas formation volume factor, bbl/scf
		self.Rsi = Rsi	# Initial gas solubility, scf/STB
		self.cw = cw	# Water compressibility, psi−1
		self.cf = cf	# Formation (rock) compressibility, psi−1
		self.Swi = Swi	# Initial water saturation

	@property
	def m(self):
		"""Ratio of initial gas-cap-gas reservoir volume to initial reservoir oil volume, bbl/bbl"""
		return (self.G*self.Bgi)/(self.N*self.Boi)

	@property
	def s(self):
		"""Storage parameter, bbl/psi"""
		return self.N*self.Boi*(1+self.m)*(self.cw*self.Swi+self.cf)/(1-self.Swi)

	def __call__(self, P, We, Np, Gp, Wp, Bo, Bw, Bg, Rs):

		self.P = P		# Volumetric average reservoir pressure
		
		self.We = We	# Cumulative water influx, bbl
		
		self.Np = Np	# Cumulative oil produced, STB
		self.Gp = Gp	# Cumulative gas produced, scf
		self.Wp = Wp	# Cumulative water produced, bbl

		self.Bo = Bo	# Oil formation volume factor, bbl/STB
		self.Bw = Bw	# Water formation volume factor, bbl/STB
		self.Bg = Bg	# Gas formation volume factor, bbl/scf
		
		self.Rs = Rs	# Gas solubility, scf/STB

		return self

	@property
	def Bt(self):
		"""Total (two-phase) formation volume factor"""
		return self.Bo+(self.Rsi-self.Rs)*self.Bg # Bti = Boi

	@property
	def A(self):
		"""A dynamic parameter used for indices calculaion."""
		return self.Np*(self.Bt+(self.Gp/self.Np-self.Rsi)*self.Bg)

	@property
	def DDI(self):
		"""depletion-drive index"""
		return self.N*(self.Bt-self.Boi)/self.A

	@property
	def SDI(self):
		"""segregation (gas-cap)-drive index"""
		return self.N*self.m*self.Boi*(self.Bg-self.Bgi)/self.Bgi/self.A

	@property
	def WDI(self):
		"""water-drive index"""
		return (self.We-self.Wp*self.Bw)/self.A

	@property
	def EDI(self):
		"""expansion (rock and liquid)-depletion index"""
		return self.s*(self.Pi-self.P)/self.A

	@property
	def total(self):
		"""The sum of the all drive indices"""
		return self.DDI+self.SDI+self.WDI+self.EDI

	def minimize(self,alter_initial=False,optimizer:dict=None,**kwargs):
		"""
		Minimizes the difference between total drive index and 1 for the current model.

		alter_initial 	: the model index whose parameters will be altered.

		Returns the OptimizeResult where the x includes the optimized values of kwargs keys.
		"""

		initial = self.initial()

		current = self.current()

		keys,values = list(kwargs.keys()),list(kwargs.values())

		def objective(values,keys,initial,current,alter_initial):

			locdict = dict(zip(keys,values))

			if alter_initial:
				initial = initial(**locdict)
				initial.update_fluid_volumes()
			else:
				current = current(**locdict)
				current.update_fluid_volumes()

			total = Cruncher.total_drive_index(initial,current)

			return (total-1)**2

		return minimize(objective,values,args=(keys,initial,current,alter_initial),**(optimizer or {})) # minimize(tank,0,N=1000_000)