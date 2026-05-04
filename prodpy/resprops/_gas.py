from ._fluid import Fluid

class GasPhase():
    """
    The hydrocarbon gases that are normally found in a natural gas are
    methanes, ethanes, propanes, butanes, pentanes, and small amounts of
    hexanes and heavier. The nonhydrocarbon gases include carbon dioxide,
    hydrogen sulfide, and nitrogen.
    
    """

    def __init__(self,*,spgr:float=None,crit=None,temp=None,zfact=None,visc=None):
        """
        Real gas defined based on specific gravity, critical properties,
            compressibility factor and/or viscosity:
        
        spgr    : specific gravity of the gas at
                  standard conditions.

        crit    : tuple of (pcrit in psi, tcrit in Rankine)

        temp    : temperature value (Rankine) which will be used to calculate
                  fluid properties when the class is called.

        zfact   : either constant value, or method to calculate z-factor
                  from pressure (psi) and temperature (Rankine).

        visc    : either constant value (cp), or method to calculate viscosity
                  from pressure (psi) and temperature (Rankine).

        Returns fluid properties viscosity, density, isothermal compressibility
            coefficient and formation volume factor when called with pressure inputs.
        """

        self._spgr  = spgr
        self._molw  = self.get_molw(spgr)/1000 # Molecular Mass, kg/mol

        pcrit,tcrit = (None,None) if crit is None else crit

        self._pcrit = None if pcrit is None else pcrit*6894.76
        self._tcrit = None if tcrit is None else tcrit*(5./9)

        self._temp  = None if temp is None else temp*(5./9)

        self.__zfact = zfact
        self.__visc  = visc

    @property
    def spgr(self):
        """specific gravity of the gas at standard conditions"""
        return self._spgr
    
    @property
    def molw(self):
        """Molecular Mass, lb/lbmol"""
        return None if self._molw else self._molw*1000

    @property
    def pcrit(self):
        """Critical Pressure in psi"""
        return None if self._pcrit is None else self._pcrit/6894.76

    @property
    def tcrit(self):
        """Critical Temperature in Rankine"""
        return None if self._tcrit is None else self._tcrit*(9./5)

    @property
    def temp(self):
        """Class temperature value in Rankine"""
        return None if self._temp is None else self._temp*(9./5)

    @property
    def tred(self):
        """Returns reduced temperature (class property)."""
        return self.temp/self.tcrit

    def pred(self,press):
        """Returns reduced pressure values for input pressure values in psi."""
        return press/self.pcrit

    def zfact(self,press):
        """Returns z-factor and (dz/dp) for given pressure values. If z is
        constant, then it return z value and 0 for (dz/dp)."""
        if callable(self.__zfact):
            return self.__zfact(press)
        return self.__zfact,0

    def visc(self,press,*args):
        """Calculates pressure and temperature dependent viscosity in cp"""
        if callable(self.__visc):
            return self.__visc(press,*args)
        return self.__visc

    def rho(self,press,zfact):
        """Returns density value in lb/ft3"""
        return (press*self.molw)/(zfact*10.731577089016*self.temp)

    def spvol(self,press,zfact):
        """Returns specific volume in ft3/lb"""
        return (zfact*10.731577089016*self.temp)/(press*self.molw)

    def comp(self,press,zfact,zprime):
        """Returns isothermal compressibility coefficient."""

        if not callable(self.__zfact):
            if self.__zfact == 0:
                return 1/press
            return

        zfact = self.__zfact

        return 1/(1+(0.27*self.pred(press))/(zfact**2*self.tred)*zprime)/press

    def fvf(self,press,zfact):
        """Returns Gas Formation Volume Factor in ft3/scf"""
        return 0.02827*zfact*self.temp/press

    def fef(self,press,zfact):
        """Returns gas expansion factor."""
        return press/(0.02827*zfact*self.temp)
    
    def __call__(self,press):
        """
        Returns Fluid instance with calculated viscosity, density, isothermal
            compressibility coefficient, and formation volume factor based on: 

        press   : Pressure values where to calculate gas properties, psi

        """
        zfact,zprime = self.zfact(press)

        visc = self.visc(press,zfact)
        rho  = self.rho(press,zfact)
        comp = self.comp(press,zfact,zprime)
        fvf  = self.fvf(press,zfact)

        return Fluid(visc,rho,comp,fvf,press)

    @staticmethod
    def get_molw(spgr):
        return None if spgr is None else spgr*28.964

    @staticmethod
    def get_spgr(molw):
        return None if molw is None else molw/28.964

if __name__ == "__main__":

    gas = Gas(5,visc=5)

    print(gas.visc)

    # fluids = gas(
    #     component=['methane','ethane'],
    #     molefraction=[0.2,0.4],
    #     moleweight=[12.5,None])

    # print(fluids.standard['pressure']['SI'])

    # print(fluids.composition['H2S'])

    # print(fluids.composition.molefraction)
    # print(fluids.composition.moleweight)
    # print(fluids.composition)