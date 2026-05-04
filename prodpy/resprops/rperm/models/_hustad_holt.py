import numpy as np

from ._stones_I import StonesI

class HustadHolt(StonesI):
    """
    This Model Provides IMBIBITION Relative Permeability MODELS for a THREE-PHASE system.

    Hustad and Holt (1992) modified Stone’s Model I by introducing anexponent term n
    to the normalized saturations.
    
    """

    def __init__(self,*args,n:float=1.,**kwargs):
        """Initializes the class.

        Parameters:
        ----------
        n : float, optional
            Saturation exponent (default is 1. which is the same as Stone's Model I)
        
        *args : tuple
            Positional arguments for the StonesI base class.

        **kwargs : dict
            Keyword arguments for the StonesI base class.
        
        """
        self.n = n

        super().__init__(*args,**kwargs)

    def kro(self,sw,so,sg,kro_ow,kro_go):
        """Returns relative oil permeability based on Hustad-Holt correlation."""
        som = self.som(sg)

        sw_star = self.sw_star(sw,som)
        so_star = self.so_star(so,som)
        sg_star = self.sg_star(sg,som)

        beta = (so_star)/(1-sw_star)/(1-sg_star)

        return (kro_ow*kro_go)/(self.k0ro_ow)*beta**self.n

    def get(self,sw:np.ndarray,so:np.ndarray,sg:np.ndarray):
        """Computes three-phase relative permeabilities krw, kro and krg.

        Parameters:
        ----------
        sw : water saturation
        so : oil saturation
        sg : gas saturation

        """
        krw,kro_ow = self.ow.get(sw)
        kro_go,krg = self.go.get(sw+so)

        kro = self.kro(sw,so,sg,kro_ow,kro_go)

        return krw,kro,krg

if __name__ == "__main__":

    pass