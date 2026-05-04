def gas_oil(method="brooks_corey",**kwargs):
	"""
	Gas-oil system relative permeability model

    sorgo   = residual oil saturation in gas-oil system
    Swc     = connate water saturation
    slc     = critical liquid saturation = Swc+Sor
    sgc     = critical gas saturation
    krogc   = oil relative permeability at critical gas saturation
    krglc   = gas relative permeability at critical liquid saturation
    no      = oil exponent on relative permeability curve
    ng      = gas exponent on relative permeability curve

	"""
    slc = self.sorgo+self.Swc
    
    movable_g = sg-self.sgc
    movable_o = 1-slc-sg
    movable_f = 1-slc-self.sgc

    krg = self.krglc*(movable_g/movable_f)**self.ng
    kro = self.krogc*(movable_o/movable_f)**self.no

    return krg,kro