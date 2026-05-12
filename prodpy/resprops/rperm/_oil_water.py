class OilWater():
	"""
	Oil-water system relative permeability model

	Sorow   = residual oil saturation in oil-water system
	Swc	    = connate water saturation
	krowc   = oil relative permeability at connate water saturaton
	krwor   = water relative permeability at the residual oil saturation
	no	    = oil exponent on relative permeability curve
	nw	    = water exponent on relative permeability curve

	"""

	def __init__(self, method="brooks_corey", **kwargs):
		pass

	def oil_water(self, sw):
		"""
		Oil-water system relative permeability model

		Sorow   = residual oil saturation in oil-water system
		Swc	    = connate water saturation
		krowc   = oil relative permeability at connate water saturaton
		krwor   = water relative permeability at the residual oil saturation
		no	    = oil exponent on relative permeability curve
		nw	    = water exponent on relative permeability curve

		"""
		slc = self.Sorow+self.Swc

		movable_w = sw-self.Swc
		movable_o = 1-slc-sw
		movable_f = 1-slc-self.Swc

		krw = self.krwor*(movable_w/movable_f)**self.nw
		kro = self.krowc*(movable_o/movable_f)**self.no

		return krw,kro