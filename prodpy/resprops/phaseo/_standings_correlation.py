import numpy as np

from ._crude_oil_system import CrudeOilSystem as cos

class StandingsCorrelation:

	@staticmethod
	def gassb_to_bpp(gassb:float,sgsg:float,gAPI:float,temp:float):
		"""
		Calculates the bubblepoint pressure of the oil at reservoir conditions
		
		Inputs:
		------
		gassb 	: Gas solubility at the bubble-point pressure, scf/STB
		
		sgsg 	: Specific gravity of the separator gas
		gAPI 	: API gravity of oil, dimensionless
		temp 	: System temperature, °F

		The equations are valid to 325F. The correlation should be used with caution
		if nonhydrocarbon components are known to be present in the system.

		"""
		x = 0.00091*temp-0.0125*gAPI

		Cpb = (gassb/sgsg)**0.83*10**x

		return 18.2*(Cpb-1.4)

	@staticmethod
	def gass_sat(p:float|np.ndarray,sgsg:float,gAPI:float,temp:float):
		"""
		Standing (1947) proposed a graphical correlation for determining the
		gas solubility as a function of pressure, gas specific gravity, API gravity,
		and system temperature.

		The correlation was developed from a total of 105 experimentally determined
		data points on 22 hydrocarbon mixtures from California crude oils and
		natural gases. The proposed correlation has an average error of 4.8%.
		
		Standing (1981) expressed his proposed graphical correlation in the
		more convenient mathematical form which requires following inputs:
		
		Inputs:
		------
		p 	 : System pressure, psia
		
		sgsg : Solution gas specific gravity
		gAPI : API gravity of oil, dimensionless
		temp : System temperature, °F

		It should be noted that Standing’s equation is valid for applications at
		and below the bubble-point pressure of the crude oil.

		"""
		x = 0.0125*gAPI-0.00091*temp

		return sgsg*((p/18.2+1.4)*10**x)**(1./0.83)

	@staticmethod
	def gass_sat_prime(p:float|np.ndarray,sgsg:float,gAPI:float,temp:float):
		"""
		Calculate the derivative of gas solubility (Rs) with respect to pressure 
		using the Standing correlation.

		The Standing method estimates the solution gas–oil ratio (Rs) in oil 
		reservoirs based on gas specific gravity, oil API gravity, reservoir 
		temperature, and pressure. This function computes:

		    dRs/dP  [scf/STB/psi]

		where:
		    Rs  = gas solubility in stock tank barrels of oil (STB)
		    P   = reservoir pressure (psi)

		Notes
		-----
		The derivative is obtained by differentiating the Standing's equation with respect 
		to p, holding gas gravity, API, and temperature constant.

		References
		----------
		Standing, M.B., 1947. 
		“A Pressure-Volume-Temperature Correlation for Mixtures of California Oils and Gases.”
		Drilling and Production Practice, API.

		Example
		-------
		>>> StandingsCorrelation.gass_sat_prime(2500, 0.85, 35, 180)
		0.0421

		"""
		gass = StandingsCorrelation.gass_sat(p,sgsg,gAPI,temp)

		return gass/(0.83*p+21.1484)

	@staticmethod
	def fvf_sat(p:float|np.ndarray,sgsg:float,gAPI:float,temp:float):
		"""
		Standing (1947) presented a graphical correlation for estimating the oil
		formation volume factor with the gas solubility, gas gravity, oil gravity,
		and reservoir temperature as the correlating parameters.

		This graphical correlation originated from examining a total of 105 experimental data
		points on 22 different California hydrocarbon systems. An average error
		of 1.2% was reported for the correlation.
		
		Standing (1981) showed that the oil formation volume factor can be expressed
		more conveniently in a mathematical form which requires following inputs:

		Inputs:
		------
		gass : Solution GOR, scf/STB
		
		sgsg : Specific gravity of the solution gas
		gAPI : API gravity of oil, dimensionless
		temp : Temperature, °T

		"""
		sgco = cos.gAPI_to_sgco(gAPI)

		gass = StandingsCorrelation.gass_sat(p,sgsg,gAPI,temp)

		CBob = gass*(sgsg/sgco)**0.5+1.25*temp

		return 0.9759+0.00012*CBob**1.2

	@staticmethod
	def fvf_sat_prime(p:float|np.ndarray,sgsg:float,gAPI:float,temp:float):
		"""
		Calculate the derivative of the oil formation volume factor (Bo) with respect 
		to pressure using the Standing correlation for saturated oil.

		The Standing correlation estimates Bo at saturation conditions based on 
		solution gas–oil ratio (Rs), gas specific gravity, oil API gravity, and 
		temperature. This method differentiates the Bo correlation with respect to 
		pressure to obtain:

		    dBo/dP  [RB/STB/psi]

		Parameters
		----------
		p : float or numpy.ndarray
		    Reservoir pressure in psi.
		sgsg : float
		    Solution gas specific gravity (air = 1.0).
		gAPI : float
		    Oil API gravity.
		temp : float
		    Reservoir temperature in °F.

		Returns
		-------
		float or numpy.ndarray
		    Derivative of oil formation volume factor with respect to pressure, 
		    dBo/dP, in RB/STB/psi.

		Notes
		-----
		Standing’s correlation for saturated oil FVF:
		    Bo = 0.9759 + 0.00012 * [ Rs * ( (γg / γo) ** 0.5 ) + 1.25 * T ] ^ 1.2

		where:
		    γo = 141.5 / (API + 131.5)
		    Rs = solution gas–oil ratio from Standing correlation.

		The derivative is computed using the chain rule:
		    dBo/dP = (∂Bo/∂Rs) * (dRs/dP)

		Both Rs and dRs/dP are evaluated from Standing's method, holding gas 
		gravity, oil gravity, and temperature constant.

		References
		----------
		Standing, M.B., 1947. 
		“A Pressure-Volume-Temperature Correlation for Mixtures of California Oils and Gases.”
		Drilling and Production Practice, API.

		Example
		-------
		>>> fvf_sat_prime(2500, 0.85, 35, 180)
		2.4e-05

		"""
		sgco = cos.gAPI_to_sgco(gAPI)

		gass = StandingsCorrelation.gass_sat(p,sgsg,gAPI,temp)

		gassp = StandingsCorrelation.gass_sat_prime(p,sgsg,gAPI,temp)

		sqrt = (sgsg/sgco)**0.5

		return 0.000144*(gass*sqrt+1.25*temp)**0.2*gassp*sqrt

if __name__ == "__main__":

	print(StandingsCorrelation.gass(250,2377+14.7,47.1,0.851))
	print(StandingsCorrelation.gass(220,2620+14.7,40.7,0.855))
	print(StandingsCorrelation.gass(260,2051+14.7,48.6,0.911))
	print(StandingsCorrelation.gass(237,2884+14.7,40.5,0.898))
	print(StandingsCorrelation.gass(218,3045+14.7,44.2,0.781))
	print(StandingsCorrelation.gass(180,4239+14.7,27.3,0.848))