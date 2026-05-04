class WyllieGardner():

	def __init__(self,swc):

		self.swc = swc # connate (irreducible) water saturation

	@sostar.setter
	def sostar(self,so):
		self._sostar = so/(1-self.swc)

	@swstar.setter
	def swstar(self,sw):
		self._swstar = (sw-self.swc)/(1-self.swc)

	@sgstar.setter
	def sgstar(self,sg):
		self._sgstar = (sg)/(1-self.swc)

	def drainage(self,sw):
		pass