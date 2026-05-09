import numpy as np

class ResInit():

	patm = 14.7

	def __init__(self,DWOC=None,DGOC=None,gradw=0.433,grado=0.346,gradg=0.043,peow=0,peog=0):
		"""
		DWOC 	: Depth of Water-Oil-Contact
		DGOC 	: Depth of Gas-Oil-Contact

		gradw 	: water hydrostatic gradient, psi/ft
		grado 	: oil hydrostatic gradient, psi/ft
		gradg 	: gas hydrostatic gradient, psi/ft

		peow 	: capillary entry pressure for oil-water
		peog 	: capillary entry pressure for oil-gas
		
		"""
		self.DWOC = DWOC
		self.DGOC = DGOC

		self.gradw = gradw
		self.grado = grado
		self.gradg = gradg

		self.peow = peow
		self.peog = peog

	def wpress(self,depth):
		"""pressure of water phase at input depth"""
		return self.pwwoc+self.gradw*(depth-self.DWOC)

	def opress(self,depth):
		"""pressure of oleic phase at input depth"""
		return self.pwwoc+self.peow+self.grado*(depth-self.DWOC)

	def gpress(self,depth):
		"""pressure of gas phase at input depth"""
		return self.pogoc+self.peog+self.gradg*(depth-self.DGOC)

	@property
	def pwwoc(self):
		"""water pressure at water-oil-contact"""
		return self.patm+self.gradw*self.DWOC

	@property
	def pwgoc(self):
		"""water pressure at gas-oil-contact"""
		return self.pwwoc+(self.DGOC-self.DWOC)*self.gradw

	@property
	def powoc(self):
		"""oil pressure at water-oil-contact"""
		return self.pwwoc+self.peow
	
	@property
	def pogoc(self):
		"""oil pressure at gas-oil-contact"""
		return self.opress(self.DGOC)

	@property
	def pggoc(self):
		"""gas pressure at gas-oil-contact"""
		return self.gpress(self.DGOC)

	@property
	def fwl(self):
		"""free-water-level"""
		return self.DWOC+self.peow/(self.gradw-self.grado)

	def saturations(self,depth,pcow,pcog=None,pcgw=None):
		"""
		Saturation values are calculated at any depth values

		depth 	: depth where to calculate saturation values
		pcow 	: oil-water drainage capillary pressure model
		pcog 	: oil-gas drainage capillary pressure model
		pcgw	: gas-water drainage capillary pressure model
		
		returns water, oil and gas saturation values.

		"""
		depth = np.asarray(depth)

		Sw = np.ones(depth.shape)
		So = np.zeros(depth.shape)
		Sg = np.zeros(depth.shape)

		zone2 = np.logical_and(depth<self.DWOC,depth>=self.DGOC)
		zone3 = depth<self.DGOC

		Sw2,So2 = self.saturations_water_oil_zone(
			depth[zone2],pcow
		)

		Sw3a,So3a,Sg3a = self.saturations_three_phase_zone(
			depth[zone3],pcow,pcog
		)

		Sw3b,Sg3b = self.saturations_water_gas_zone(
			depth[zone3],pcgw
		)

		Sw3 = np.zeros(Sw3a.shape)
		So3 = np.zeros(So3a.shape)
		Sg3 = np.zeros(Sg3b.shape)

		Sw3[So3a>=0] = Sw3a[So3a>=0]
		So3[So3a>=0] = So3a[So3a>=0]
		Sg3[So3a>=0] = Sg3a[So3a>=0]

		Sw3[So3a<0] = Sw3b[So3a<0]
		So3[So3a<0] = 0
		Sg3[So3a<0] = Sg3b[So3a<0]

		Sw[zone2] = Sw2
		Sw[zone3] = Sw3

		So[zone2] = So2
		So[zone3] = So3

		Sg[zone3] = Sg3

		return Sw,So,Sg

	def saturations_water_oil_zone(self,depth,pcow):
		"""Saturation values are calculated assuming that
		the existing phases are water and oil.
		
		depth 	: depth where to calculate saturation values
		pcow 	: oil-water capillary pressure model

		returns water and oil saturation values

		"""
		depth = np.asarray(depth)

		pw = self.wpress(depth)
		po = self.opress(depth)

		Sw = pcow.idrainage(po-pw)

		return Sw,1-Sw

	def saturations_three_phase_zone(self,depth,pcow,pcog):
		"""saturation values are calculated assuming that
		there are three existing phases: water, oil, and gas.

		depth 	: depth where to calculate saturation values
		pcow 	: oil-water capillary pressure model
		pcog 	: oil-gas capillary pressure model

		returns water, oil, and gas saturation values.

		"""
		depth = np.asarray(depth)

		pw = self.wpress(depth)
		po = self.opress(depth)
		pg = self.gpress(depth)

		Sl = pcog.idrainage(pg-po)
		Sw = pcow.idrainage(po-pw)

		return Sw,Sl-Sw,1-Sl

	def saturations_water_gas_zone(self,depth,pcgw):
		"""saturation values are calculated assuming that
		there are two phases: water and gas.
		
		depth 	: depth where to calculate saturation values
		pcgw	: gas-water capillary pressure model

		returns water and gas saturation values.
		
		"""
		depth = np.asarray(depth)

		pw = self.wpress(depth)
		pg = self.gpress(depth)

		Sw = pcgw.idrainage(pg-pw)

		return Sw,1-Sw


