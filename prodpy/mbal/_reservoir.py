from dataclasses import dataclass

import logging

import numpy

@dataclass
class Reservoir:
	"""VOLUMETRIC AVERAGE PARAMETERS FOR MATERIAL BALANCE TANK

	P 		: Volumetric average reservoir pressure
			  at which fluid and rock properties are defined, psi

	Sw 		: Water saturation,

	N 		: Oil in place, STB
	G		: Gas-cap gas, scf

	We		: Cumulative water influx, bbl

	"""

	P 		: float = None
	Sw 		: float = None

	N 		: float = None
	G 		: float = None

	We 		: float = 0.

	@staticmethod
	def get(**kwargs):
		return {key: value for key, value in kwargs.items() if key in Reservoir().__dict__}

if __name__ == "__main__":

	res = Reservoir()

	print(res.get(a=5,d=5,P=8))

	print(res.__dict__)

	print(Reservoir().__dict__)

	print(Reservoir.get(a=5,d=5,P=8))