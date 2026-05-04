import numpy as np

from ._crude_oil_system import CrudeOilSystem as cos

class VasquezBeggsCorrelation:

	@staticmethod
	def sgsg_corr(sgsg:float,gAPI:float,psep:float=None,Tsep:float=None):
		"""
		Method to calculate corrected specific gravity for the solution gas.

		Realizing that the value of the specific gravity of the gas depends on
		the conditions under which it is separated from the oil, Vasquez and
		Beggs proposed that the value of the gas specific gravity as obtained
		from a separator pressure of 100 psig be used in the fluid property
		calculation equations. This reference pressure was chosen because it
		represents the average field separator conditions.
		
		Inputs:
		------
		sgsg  	: Gas gravity at the actual separator conditions of psep and Tsep
		gAPI	: API oil gravity
		psep	: Actual separator pressure, psia
		Tsep	: Actual separator temperature, °F

		Returns:
		-------
		Gas gravity at the reference separator pressure

		The gas gravity used to develop all the correlations reported by the
		authors was that which would result from a two-stage separation. The
		first-stage pressure was chosen as 100 psig and the second stage was the
		stock tank.

		"""
		if psep and Tsep:
			return sgsg*(1+5.912e-5*gAPI*Tsep*np.log10(psep/(100+14.7)))

		return sgsg

	@staticmethod
	def gassb_to_bpp(gassb:float,sgsg:float,gAPI:float,temp:float,psep:float=None,Tsep:float=None):
		"""
		Vasquez and Beggs’ gas solubility correlation can be solved for the
		bubble-point pressure that will require the following inputs:
		
		Inputs:
		------
		gassb 	: Gas solubility at the bubble-point pressure, scf/STB
		
		sgsg 	: Specific gravity of the separator gas
		gAPI 	: API oil gravity, dimensionless
		temp 	: System temperature, °F
		
		psep 	: Separator pressure, psia
		Tsep 	: Separator temperature, °F

		"""
		if gAPI<=30.:
			C1 = 27.624
			C2 = 0.914328
			C3 = 11.172
		else:
			C1 = 56.18
			C2 = 0.84246
			C3 = 10.393

		sgsg = VasquezBeggsCorrelation.sgsg_corr(sgsg,gAPI,psep,Tsep)
		
		a = -C3*gAPI/(temp+460)

		return (C1*gassb/sgsg*10**a)**C2

	@staticmethod
	def gass_sat(p:float|np.ndarray,sgsg:float,gAPI:float,temp:float,psep:float=None,Tsep:float=None):
		"""
		Vasquez and Beggs (1980) presented an improved empirical correlation
		for estimating Rs. The correlation was obtained by regression analysis
		using 5,008 measured gas solubility data points.

		Based on oil gravity, the measured data were divided into two groups.
		This division was made at a value of oil gravity of 30°API.
		
		Inputs:
		------
		p		: System pressure, psia
		
		sgsg	: Solution gas specific gravity
		gAPI 	: API gravity of oil, dimensionless
		temp	: System temperature, °F
		
		psep	: Separator pressure, psia
		Tsep	: Separator temperature, °F

		An independent evaluation of the above correlation by Sutton and
		Farashad (1984) shows that the correlation is capable of predicting gas
		solubilities with an average absolute error of 12.7%.

		"""
		if gAPI<=30.:
			C1 = 0.0362
			C2 = 1.0937
			C3 = 25.7240
		else:
			C1 = 0.0178
			C2 = 1.1870
			C3 = 23.931

		sgsg = VasquezBeggsCorrelation.sgsg_corr(sgsg,gAPI,psep,Tsep)

		return C1*sgsg*p**C2*np.exp(C3*gAPI/(temp+460))

	@staticmethod
	def gass_sat_prime(p:float|np.ndarray,sgsg:float,gAPI:float,temp:float,psep:float=None,Tsep:float=None):
		"""
		Calculate the derivative of gas solubility (Rs) with respect to pressure 
		using the Vasquez-Beggs correlation.

		The method estimates the solution gas–oil ratio (Rs) in oil 
		reservoirs based on gas specific gravity, oil API gravity, reservoir 
		temperature, and pressure. This function computes:

		    dRs/dP  [scf/STB/psi]

		where:
		    Rs  = gas solubility in stock tank barrels of oil (STB)
		    P   = reservoir pressure (psi)

		Notes
		-----
		The derivative is obtained by differentiating the Vasquez-Beggs equation with respect 
		to p, holding gas gravity, API, and temperature constant.

		Example
		-------
		>>> VasquezBeggsCorrelation.gass_sat_prime(2500, 0.85, 35, 180)
		0.0421

		"""
		C2 = 1.0937 if gAPI<=30. else 1.1870

		return C2/p*VasquezBeggsCorrelation.gass_sat(p,sgsg,gAPI,temp,psep,Tsep)

	@staticmethod
	def fvf_sat(p:float|np.ndarray,sgsg:float,gAPI:float,temp:float,psep:float=None,Tsep:float=None):
		"""
		Calculates Oil Formation Volume Factor in bbl/stb

		Vasquez and Beggs (1980) developed a relationship for determining
		Bo as a function of Rs, sgco, sgsg, and temp. The proposed correlation was based
		on 6,000 measurements of Bo at various pressures. Using the regression
		analysis technique, Vasquez and Beggs found the method to
		be the best form to reproduce the measured data:
		
		Inputs:
		------
		p	 : System pressure, psia
		
		sgsg : gas specific gravity
		gAPI : API oil gravity
		temp : temperature, °F
		
		psep : separator pressure, psia
		Tsep : separator temperature, °F

		Vasquez and Beggs reported an average error of 4.7% for the proposed
		correlation.

		"""
		if gAPI<=30.:
			C1 = 4.677E-04
			C2 = 1.751E-05
			C3 = -1.811E-08
		else:
			C1 = 4.670E-04
			C2 = 1.100E-05
			C3 = 1.337E-09

		sgsg = VasquezBeggsCorrelation.sgsg_corr(sgsg,gAPI,psep,Tsep)

		Rs = VasquezBeggsCorrelation.gass_sat(p,sgsg,gAPI,temp)
		
		return 1.+C1*Rs+(temp-60)*(gAPI/sgsg)*(C2+C3*Rs)

	@staticmethod
	def fvf_sat_prime(p:float|np.ndarray,sgsg:float,gAPI:float,temp:float,psep:float=None,Tsep:float=None):
		"""
		Calculate the derivative of the oil formation volume factor (Bo) with respect 
		to pressure using the Vasquez-Beggs correlation for saturated oil.

		The correlation estimates Bo at saturation conditions based on 
		solution gas–oil ratio (Rs), gas specific gravity, oil API gravity, and 
		temperature. This method differentiates the Bo correlation with respect to 
		pressure to obtain:

		    dBo/dP  [RB/STB/psi]

		Parameters
		----------
		p 	 : float or numpy.ndarray
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

		The derivative is computed using the chain rule:
		    dBo/dP = (∂Bo/∂Rs) * (dRs/dP)

		Both Rs and dRs/dP are evaluated from Vasquez-Beggs method, holding gas 
		gravity, oil gravity, and temperature constant.

		Example
		-------
		>>> fvf_sat_prime(2500, 0.85, 35, 180)
		2.4e-05

		"""
		C1 =  4.677E-04 if gAPI<=30. else 4.670E-04
		C3 = -1.811E-08 if gAPI<=30. else 1.337E-09

		Rsp = VasquezBeggsCorrelation.gass_sat_prime(p,sgsg,gAPI,temp,psep,Tsep)

		return C1*Rsp+(temp-60)*(gAPI/sgsg)*(C3*Rsp)

	@staticmethod
	def fvf_nonsat(p:float|np.ndarray,bpp:float,sgsg:float,gAPI:float,temp:float,psep:float=None,Tsep:float=None):
		"""
		Calculate the undersaturated oil formation volume factor (FVF) 
		using the Vasquez–Beggs correlation.

		This method estimates the oil FVF at pressures above the bubble point 
		by first computing the saturated FVF at the bubble point and then 
		applying the Vasquez–Beggs pressure correction for undersaturated conditions.
		If separator gas specific gravity is provided at non-standard conditions, 
		it is corrected to stock-tank conditions before calculations.

		Parameters
		----------
		p : float or np.ndarray
		    Reservoir pressure(s) [psia].
		bpp : float
		    Bubble point pressure [psia].
		sgsg : float
		    Gas specific gravity at separator conditions (air = 1.0).
		gAPI : float
		    Oil API gravity.
		temp : float
		    Reservoir temperature [°F].
		psep : float, optional
		    Separator pressure [psia]. Used for gas specific gravity correction.
		Tsep : float, optional
		    Separator temperature [°F]. Used for gas specific gravity correction.

		Returns
		-------
		float or np.ndarray
		    Undersaturated oil formation volume factor, Bo [rb/STB].

		Notes
		-----
		- If `p <= bpp`, the returned value corresponds to the saturated FVF at bubble point.
		- Gas specific gravity is corrected to stock-tank conditions if `psep` and `Tsep` are provided.
		- Correlation source: Vasquez, M.E., and Beggs, H.D. (1980).
		  “Correlations for Fluid Physical Property Prediction.”
		  Journal of Petroleum Technology, 32(6), 968–970.

		Examples
		--------
		>>> VasquezBeggsCorrelation.fvf_nonsat(3500, 3000, 0.85, 35, 180)
		1.2428

		"""
		p = np.where(p<bpp,np.nan,p)

		sgsg = VasquezBeggsCorrelation.sgsg_corr(sgsg,gAPI,psep,Tsep)

		Bob = VasquezBeggsCorrelation.fvf_sat(bpp,sgsg,gAPI,temp)
		Rsb = VasquezBeggsCorrelation.gass_sat(bpp,sgsg,gAPI,temp)

		A = 1e-5*(-1433.+5.*Rsb+17.2*temp-1180.*sgsg+12.61*gAPI)

		return Bob*np.exp(-A*np.log(p/bpp))

	@staticmethod
	def comp_nonsat(p:float|np.ndarray,bpp:float,sgsg:float,gAPI:float,temp:float,psep:float=None,Tsep:float=None):
		"""Calculates oil isothermal compressibility in 1/psi for pressures above
		buble point.

		From a total of 4,036 experimental data points used in a linear regression
		model, Vasquez and Beggs (1980) correlated the isothermal oil compressibility
		coefficients with Rsb, T, °API, gg, and p. They proposed the
		following expression:
		
		Inputs:
		------
		p  	 : pressure, psia

		bpp  : bubble point pressure [psia].
		
		sgsg : gas specific gravity
		gAPI : API oil gravity
		temp : temperature, °F
		
		psep : separator pressure, psia
		Tsep : separator temperature, °F

		"""
		p = np.where(p<bpp,np.nan,p)

		sgsg = VasquezBeggsCorrelation.sgsg_corr(sgsg,gAPI,psep,Tsep)

		Rsb = VasquezBeggsCorrelation.gass_sat(bpp,sgsg,gAPI,temp)

		return 1e-5*(-1433.+5.*Rsb+17.2*temp-1180.*sgsg+12.61*gAPI)/p
		
if __name__ == "__main__":

	print(VasquezBeggsCorrelation.gass(250,2377,47.1,0.851, 60,150+14.7))
	print(VasquezBeggsCorrelation.gass(220,2620,40.7,0.855, 75,100+14.7))
	print(VasquezBeggsCorrelation.gass(260,2051,48.6,0.911, 72,100+14.7))
	print(VasquezBeggsCorrelation.gass(237,2884,40.5,0.898,120, 60+14.7))
	print(VasquezBeggsCorrelation.gass(218,3045,44.2,0.781, 60,200+14.7))
	print(VasquezBeggsCorrelation.gass(180,4239,27.3,0.848,173, 85+14.7))

	print(VasquezBeggsCorrelation.sgsg_corr( 60,150+14.7,47.1,0.851))
	print(VasquezBeggsCorrelation.sgsg_corr( 75,100+14.7,40.7,0.855))
	print(VasquezBeggsCorrelation.sgsg_corr( 72,100+14.7,48.6,0.911))
	print(VasquezBeggsCorrelation.sgsg_corr(120, 60+14.7,40.5,0.898))
	print(VasquezBeggsCorrelation.sgsg_corr( 60,200+14.7,44.2,0.781))
	print(VasquezBeggsCorrelation.sgsg_corr(173, 85+14.7,27.3,0.848))