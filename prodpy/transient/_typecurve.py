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

	