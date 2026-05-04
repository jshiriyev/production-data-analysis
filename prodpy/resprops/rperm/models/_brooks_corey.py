import numpy as np

class BrooksCorey():
    """
    Implements the Brooks-Corey relative permeability model (1964) for two-phase flow 
    (for wetting [w] and non-wetting [o] phases).

    """

    def __init__(self,swr:float=0.2,sor:float=0.2,k0rw:float=0.2,k0ro:float=0.2,nw:float=2.,no:float=2.):
        """Initializes the class.

        Parameters:
        ----------
        swr   : float, residual wetting phase saturation.
        sor   : float, residual non-wetting phase saturation.
        k0rw  : float, end-point wetting phase relative permeability (at sor).
        k0ro  : float, end-point non-wetting phase relative permeability (at swr).
        nw    : float, wetting phase exponent.
        no    : float, non-wetting phase exponent.

        """
        self.swr = swr
        self.sor = sor

        self.k0rw = k0rw
        self.k0ro = k0ro

        self.nw = nw
        self.no = no

    def swd(self,sw):
        """Returns dimensionless water saturation (starred)."""
        values = (np.ravel(sw)-self.swr)/(1-self.swr-self.sor)

        return np.clip(values,0,1)

    def get(self,sw:np.ndarray):
        """Computes relative permeabilities krw and kro at given wetting phase saturation, sw.

        Parameters:
        ----------
        sw  : np.ndarray, wetting phase saturation.

        Returns:
        -------
        krw : np.ndarray, water relative permeability.
        kro : np.ndarray, oil relative permeability.

        """
        S = self.swd(sw)
        
        krw = self.k0rw*S**self.nw
        kro = self.k0ro*(1-S)**self.no

        return krw,kro

if __name__ == "__main__":

    rp = BrooksCorey(0.1,0.4,0.3,0.8)

    print(rp.get(0.3))

    rp = BrooksCorey(0.3,0.05,0.8,0.3)

    print(rp.get(0.8))