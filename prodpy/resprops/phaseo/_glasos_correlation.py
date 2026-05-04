import numpy as np

from ._crude_oil_system import CrudeOilSystem as cos

class GlasosCorrelation:

	@staticmethod
	def gassb_to_bpp(Rsb:float,sgsg:float,gAPI:float,temp:float):
		pass

	@staticmethod
	def gass_sat(p:float|np.ndarray,bpp:float,sgsg:float,gAPI:float,temp:float):
		"""
		Glaso (1980) proposed a correlation for estimating the gas solubility as
		a function of the API gravity, pressure, temperature, and gas specific gravity.
		The correlation was developed from studying 45 North Sea crude oil
		samples. Glaso reported an average error of 1.28% with a standard deviation
		of 6.98%.

		p 	 : system pressure, psia

		bpp  : Bubble point pressure, psia

		sgsg : solution gas specific gravity
		gAPI : API gravity, dimensionless
		temp : temperature, °F

		"""
		xb = 2.8869-(14.1811-3.3093*np.log10(bpp))**0.5

		Rsb = sgsg*(gAPI**0.989/temp**0.172*(10**xb))**1.2255

		p = np.atleast_1d(p)

		x = 2.8869-(14.1811-3.3093*np.log10(p[p<bpp]))**0.5

		_Rs = np.full_like(p,Rsb)
		_Rs[p<bpp] = sgsg*(gAPI**0.989/temp**0.172*(10**x))**1.2255

		return _Rs

	@staticmethod
	def gass_sat_prime(p:float|np.ndarray,sgsg:float,gAPI:float,temp:float):
		pass

	@staticmethod
	def fvf_sat(p:float|np.ndarray,sgsg:float,gAPI:float,temp:float):
		pass

	@staticmethod
	def fvf_sat_prime(p:float|np.ndarray,sgsg:float,gAPI:float,temp:float):
		pass

	@staticmethod
	def fvf_nonsat(p:float|np.ndarray,bpp:float,sgsg:float,gAPI:float,temp:float):
		pass

	@staticmethod
	def comp_sat(p:float|np.ndarray):
		pass

	@staticmethod
	def comp_nonsat(p:float|np.ndarray,bpp:float,fvfg:np.ndarray,fvfo:np.ndarray,sgsg:float,gAPI:float,temp:float):
		pass

if __name__ == "__main__":

	print(glaso(250,2377+14.7,47.1,0.851))
	print(glaso(220,2620+14.7,40.7,0.855))
	print(glaso(260,2051+14.7,48.6,0.911))
	print(glaso(237,2884+14.7,40.5,0.898))
	print(glaso(218,3045+14.7,44.2,0.781))
	print(glaso(180,4239+14.7,27.3,0.848))