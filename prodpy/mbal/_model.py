import copy

from ._reservoir import Reservoir
from ._phase import Phase
from ._operation import Operation
from ._cruncher import Cruncher

class Model():

	def __init__(self,M=None,**kwargs):
		"""Initialization of Material Balance Tank Model."""

		self._M = M

		self._reservoir  = Reservoir(**Reservoir.get(**kwargs))
		self._phase      = Phase(**Phase.get(**kwargs))
		self._operation  = Operation(**Operation.get(**kwargs))

		self.update_fluid_volumes()

	@property
	def reservoir(self):
		return self._reservoir

	@property
	def phase(self):
		return self._phase

	@property
	def operation(self):
		return self._operation

	def __call__(self,inplace=False,**kwargs):
		"""Updating Material Balance Tank Model properties."""

		if not inplace:
			self = copy.deepcopy(self)

		for key,value in kwargs.items():

			if key in self.reservoir.__dict__:
				setattr(self.reservoir,key,value)
			elif key in self.phase.__dict__:
				setattr(self.phase,key,value)
			elif key in self.operation.__dict__:
				setattr(self.operation,key,value)
			else:
				print(f"Warning: Key '{key}' not found in any sub-class properties.")

		if not inplace:
			return self

	def update_fluid_volumes(self):

		if self.phase.Bo is None:
			return

		if self.phase.Bg is None:
			return

		if self.reservoir.N is not None and self._M is not None:
			self.reservoir.G = Cruncher.G(self,self._M)
		elif self.reservoir.G is not None and self._M is not None:
			self.reservoir.N = Cruncher.N(self,self._M)
		elif self.reservoir.N is not None and self.reservoir.G is not None:
			self._M = Cruncher.M(self)

	@property
	def N(self):
		return self.reservoir.N
	
	@N.setter
	def N(self,value):
		self.reservoir.N = value
		self.update_fluid_volumes()

	@property
	def G(self):
		return self.reservoir.G

	@G.setter
	def G(self,value):
		self.reservoir.G = value
		self.update_fluid_volumes()

	@property
	def M(self):
		"""Ratio of gas-cap-gas reservoir volume to reservoir oil volume, bbl/bbl"""
		return self._M

	@M.setter
	def M(self,value):
		self._M = value
		self.update_fluid_volumes()

	@property
	def PV(self):
		"""Total pore volume filled with hydrocarbon, bbl"""
		return Cruncher.PV(self)
	