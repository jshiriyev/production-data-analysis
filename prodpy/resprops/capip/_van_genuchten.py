import numpy as np

from ._capillary_pressure import BaseClass

class VanGenuchten(BaseClass):

    def __init__(self,swr:float=0,sor:float=0,n:float=2,m:float=2,gamma:float=0.02):
        """
        The Van Genuchten (1980) model.

        Parameters:
        ----------
        swr    : irreducible water saturation
        sor    : residual oil saturation

        n, m, gamma   : empirical constants

        """
        self.swr = swr
        self.sor = sor

        self.n = n
        self.m = m

        self.gamma = gamma

    @property
    def im(self):
        """Returns an instance of the Imbibition class."""
        return Imbibition(self.swr,self.sor,self.n,self.m,self.gamma)

class Imbibition(VanGenuchten):

    def __init__(self,*args,**kwargs):

        super().__init__(*args,**kwargs)

    def se(self,sw:np.ndarray):
        """Calculates non-dimensional saturation values."""
        sd = (np.ravel(sw)-self.swr)/(1-self.swr-self.sor)
        sd = np.clip(sd,0,1)

        return sd

    def pc(self,sw):
        """Calculates capillary pressure from saturation."""
        se = self.se(sw)
        pc = np.empty_like(se)

        pc[se==0] = np.inf
        pc[se!=0] = 1/self.gamma*(se[se!=0]**(-1/self.m)-1)**(1/self.n)

        return pc

    def jf(self,sw:np.ndarray):
        """Return J-function that is function to calculate J-function values."""
        return lambda *args,**kwargs: self.pc2jf(self.pc(sw),*args,**kwargs)

    def sw(self,pc):
        """Calculates saturation from capillary pressure, inverse calculations."""
        se = ((self.gamma*pc)**self.n+1)**(-self.m)

        return self.swr+se*(1-self.swr-self.sor)