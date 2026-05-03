class TypeCurve():

	def __init__(self,rrock,phase,wcond,immob=None,tcomp=None):
		"""
		rrock 	: reservoir rock instance
		phase 	: fluid phase instance
		wcond 	: well instance
		immob 	: immobile second phase instance
		tcomp 	: total compressibility
		
		"""
		self.rrock = rrock
		self.phase = phase
		self.tcomp = tcomp
		self.wcond = wcond
		self.immob = immob

	def set_time(self,*args):

		self.time  = Time(*args)

	def init(self,pzero):

		self.pzero = pzero

	def press(self):

		pass

	def tD(self):

		return (self.diffuse*self.time.times)/(self.wcond.radius**2)*0.00633

	def CD(self):

		vpore = self.rrock.poro*self.rrock.height*self.wcond.radius**2

		return (self.storage)/(vpore*self.ccomp)*0.8936

	def pD(self,press):

		return (self.pzero-press)/self.pcons*141.2

	@property
	def ccomp(self):
		"""Returns the total compressibility, self.tcomp. If it is None, the function will
		return calculated total compressibility from rock and fluid compressibilities."""

		if self.tcomp is not None:
			return self.tcomp

		comp2,satur2 = 0.,0.

		if self.immob is not None:
			comp2,satur2 = self.immob.comp,self.immob.satur

		return self.rrock.comp+self.phase.comp*(1-satur2)+comp2*satur2

	@property
	def diffuse(self):
		return (self.rrock.perm)/(self.rrock.poro*self.fluid.visc*self.ccomp)

	@property
	def storage(self):
		return (self.wcond.area)/(self.fluid.rho)*144/5.615

	@property
	def pcons(self):
		return (self.wcond.rate)/(self.fluid.mobil*self.rrock.perm*self.rrock.height)*141.2