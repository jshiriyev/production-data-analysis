import numpy as np

class GasOilInterfacialTension:
	
	def tens(p:np.ndarray,gAPI:float,temp:float):
		"""Function to Calculate Gas-Oil Interfacial Tension in dynes/cm

		p      : pressure, psia

		gAPI   : API oil gravity
		temp   : temperature, °F

		"""
		s068 = 39.0 - 0.2571 * gAPI
		s100 = 37.5 - 0.2571 * gAPI

		if (temp <= 68):
		    st = s068
		elif (temp >= 100):
		    st = s100
		else:
		    st = s068 - (temp - 68) * (s068 - s100) / 32

		c = 1 - 0.024 * p ** 0.45
		so = c * st
		
		if so < 1:
		    so = 1

		return so