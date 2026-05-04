import logging

import numpy as np

from scipy import special

from ._solver_object import SolverObj
from ._radial_pormed import RadPorMed

from ._result import Result

class TransientState(RadPorMed,SolverObj):
    """
    Transient solution of the diffusivity equation in radial coordinates using 
    the line source solution based on the exponential integral.

    Inherits from:
        RadPorMed: Provides radial porous media properties.
        SolverObj: Handles base solver configurations and behaviors.

    """

    def __init__(self,size:tuple,**kwargs):
        """
        Initializes the Transient solver by invoking the initializers of 
        RadPorMed and SolverObj.

        Args:
            size (float tuple): porous media size for RadPorMed.
            **kwargs: Keyword arguments for SolverObj.

        """
        RadPorMed.__init__(self,size)
        SolverObj.__init__(self,**kwargs)

        self.tmax = None

    @property
    def vpore(self):
        """Getter for the pore volume."""
        if not hasattr(self,"_vpore"):
            self.vpore = None

        return self._vpore/(0.3048**3)

    @vpore.setter
    def vpore(self,value):
        """Setter for the pore volume."""
        self._vpore = self._volume*self.layer._poro

    def __call__(self,well,pinit:float=None):
        """
        Prepares the Transient solver instance for execution by configuring 
        the well and optional initial pressure.

        Args:
            well: An object representing the well to be simulated.
            pinit (float, optional): Initial reservoir pressure. Defaults to None.

        Returns:
            TransientState: Returns self to allow for method chaining or deferred execution.
        
        """
        self.well = well
        self.tmin = None
        self.term = None

        self.pinit = pinit

        return self

    @property
    def well(self):
        """Getter for the well item."""
        return self._well
    
    @well.setter
    def well(self,value):
        """Setter for the well item."""
        self._well = value

    @property
    def tmin(self):
        """Getter for the minimum time limit for the transient solution because of the finite wellbore size."""
        return self._tmin/(24*60*60)
    
    @tmin.setter
    def tmin(self,value):
        """Setter for the mimimum time limit for the transient solution because of the finite wellbore size."""
        self._tmin = 100*self.well._radius**2/self._hdiff

    @property
    def tmax(self):
        """Getter for the maximum time limit for the transient solution because of the finite reservoir size."""
        return self._tmax/(24*60*60)

    @tmax.setter
    def tmax(self,tmax=None):
        """Setter for the maximum time limit for the transient solution because of the finite reservoir size."""
        self._tmax = 0.25*self._radius**2/self._hdiff

    @property
    def term(self):
        """Getter for the pressure term used in analytical equations."""
        return self._term/6894.76

    @term.setter
    def term(self,value):
        """Setter for the pressure term used in analytical equations."""
        self._term = (self.well._cond)/(2*np.pi*self._flow*self.fluid._mobil)

    @property
    def pinit(self):
        """Getter for the initial reservoir pressure."""
        return self._pinit/6894.76

    @pinit.setter
    def pinit(self,values):
        """Setter for the initial reservoir pressure."""
        self._pinit = np.ravel(value).astype(float)*6894.76

    def solve(self,times,nodes):
        """Solves for the pressure values at transient state."""
        result = Result(self.correct(times),nodes)

        eiterm = special.expi(-(result._nodes**2)/(4*self._hdiff*result._times))

        result._press = self._pinit-self._term*(-1/2*eiterm+self.well._skin)

        return result

    def correct(self,times:np.ndarray):
        """It sets the time values to np.nan if it is outside of the solver limit."""
        bound_internal = times>=self.tmin
        bound_external = times<=self.tmax

        if np.any(~bound_internal):
            logging.warning("Not all times satisfy the early time limits!")

        if np.any(~bound_external):
            logging.warning("Not all times satisfy the late time limits!")

        valids = np.logical_and(bound_internal,bound_external)

        return np.where(valids,times,np.nan)

if __name__ == "__main__":

    t = Transient(2,2,layer="None")