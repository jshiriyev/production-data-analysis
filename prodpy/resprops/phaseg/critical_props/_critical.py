import json

with open("_elms.json") as elmlib:
    elements = json.load(elmlib)

class critic:
	"""
	Static method collection for critical property calculation based
	on the specific gravity of the gas.
	
	"""

	@staticmethod
    def get_mwa(mfracs,mweight):
        """Calculates apparent molecular weight."""
        return sum([frac*weight for frac,weight in zip(mfracs,mweight)])

    @staticmethod
    def get_spgr_atsc(mwa):
        """The calculation assumes that the behavior of both the gas mixture and
        air is described by the ideal gas equation at standard conditions.

        """
        return mwa/28.964

	@staticmethod
	def pseudo(mfracs,pcrits,tcrits):
	    """It calculates pseudo-critical temperature and pressure based on
	    mole fraction and pseudo properties of each component.

	    """
	    ppcs = [yi*Pci for yi,Pci in zip(mfracs,pcrits)]
	    tpcs = [yi*Tci for yi,Tci in zip(mfracs,tcrits)]

	    return sum(ppcs),sum(tpcs)
	
	@staticmethod
	def natural(spgr):
		"""It calculates pseudo-critical temperature and pressure for
		natural-gas systems based on specific gravity:

		spgr    : specific gravity of the gas at
                  standard conditions.

        Output:

		ppc     : Critical Pressure in psia
		tpc     : Critical Temperature in °R

		"""
		ppc = 677+15*spgr-37.5*spgr**2
		tpc = 168+325*spgr-12.5*spgr**2

		return ppc,tpc

	@staticmethod
	def condensate(spgr):
		"""It calculates pseudo-critical temperature and pressure for
		gas-condensate systems based on specific gravity:

		spgr    : specific gravity of the gas at
                  standard conditions.

        Output:

		ppc     : Critical Pressure in psia
		tpc     : Critical Temperature in °R

		"""
		ppc = 706-51.7*spgr-11.1*spgr**2
		tpc = 187+330.0*spgr-71.5*spgr**2

		return ppc,tpc

	@staticmethod
	def highweight(spgr):
		"""It calculates pseudo-critical properties for high molecular
		weight reservoir gases based on specific gravity:

		spgr    : specific gravity of the gas at
                  standard conditions.

        Output:

		ppc     : Critical Pressure in psia
		tpc     : Critical Temperature in °R

		"""
		ppc = 756.8-131*spgr-3.6*spgr**2
		tpc = 169.2+349.5*spgr-74*spgr**2

		return ppc,tpc

if __name__ == "__main__":

	print(critic.natural(0.8))
	print(critic.condensate(0.8))
	print(critic.highweight(0.8))