import numpy as np

from ._capillary_pressure import BaseClass

class BrooksCorey(BaseClass):

    def __init__(self,swr:float=0.,sor:float=0.,lamda:float=2.,entry:float=4.5):
        """
        BrooksCorey capillary  pressure model.

        Parameters:
        ----------
        swr     : irreducible water saturation
        sor     : residual oil saturation

        lamda   : empirical model constant
        entry   : capillary entry pressure

        """
        self.swr = swr
        self.sor = sor

        self.lamda = lamda
        self.entry = entry

    @property
    def dr(self):
        """Returns an instance of the Drainage class."""
        return Drainage(self.swr,self.sor,self.lamda,self.entry)

    @property
    def im(self):
        """Returns an instance of the Imbibition class."""
        return Imbibition(self.swr,self.sor,self.lamda,self.entry)

class Drainage(BrooksCorey):
    """
    Drainage process in the Brooks-Corey model.

    """
    def __init__(self,*args,**kwargs):

        super().__init__(*args,**kwargs)

    def ss(self,sw:np.ndarray):
        """Calculates non-dimensional saturation values."""
        sd = (np.ravel(sw)-self.swr)/(1-self.swr)
        sd = np.clip(sd,0,1)

        return sd

    def pc(self,sw:np.ndarray):
        """Calculates capillary pressure from wetting phase saturation."""
        ss = self.ss(sw)

        pc = np.empty_like(ss)

        pc[ss==0] = np.inf
        pc[ss!=0] = self.entry*ss[ss!=0]**(-1/self.lamda)

        return pc

    def jf(self,sw:np.ndarray):
        """Return J-function that is function to calculate J-function values."""
        return lambda *args,**kwargs: self.pc2jf(self.pc(sw),*args,**kwargs)

    def sw(self,pc:np.ndarray):
        """Calculates saturation from capillary pressure, inverse calculations."""
        ss = (self.entry/np.ravel(pc))**(self.lamda)

        return self.swr+ss*(1-self.swr)

class Imbibition(BrooksCorey):
    """
    Imbibition process in the Brooks-Corey model.

    """
    def __init__(self,*args,**kwargs):

        super().__init__(*args,**kwargs)

    def se(self,sw:np.ndarray):
        """Calculates non-dimensional saturation values."""
        sd = (np.ravel(sw)-self.swr)/(1-self.swr-self.sor)
        sd = np.clip(sd,0,1)

        return sd

    def pc(self,sw:np.ndarray):
        """Calculates capillary pressure from saturation."""
        se = self.se(sw)
        pc = np.empty_like(se)

        pc[se==0] = np.inf
        pc[se!=0] = self.entry*(se[se!=0]**(-1/self.lamda)-1)

        return pc

    def jf(self,sw:np.ndarray):
        """Return J-function that is function to calculate J-function values."""
        return lambda *args,**kwargs: self.pc2jf(self.pc(sw),*args,**kwargs)

    def sw(self,pc:np.ndarray):
        """Calculates saturation from capillary pressure, inverse calculations."""
        se = (np.ravel(pc)/self.entry+1)**(-self.lamda)

        return self.swr+se*(1-self.swr-self.sor)