import numpy as np

from ._solver_object import SolverObj
from ._radial_pormed import RadPorMed

from ._result import Result

class SteadyState(RadPorMed,SolverObj):
    """
    Steady state solution of the diffusivity equation in radial coordinates using 
    the line source solution based on the exponential integral.

    Inherits from:
        RadPorMed: Provides radial porous media properties.
        SolverObj: Handles base solver configurations and behaviors.

    """

    def __init__(self,size:tuple,**kwargs):
        """Initializes steady state constant rate radial solution of
        diffusivity equation.

        Args:
            size (float tuple): porous media size for RadPorMed.
            **kwargs: Keyword arguments for SolverObj.

        """
        RadPorMed.__init__(self,size)
        SolverObj.__init__(self,**kwargs)

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
        Prepares the Steady-State solver instance for execution by configuring 
        the well and optional shape and initial pressure.

        Args:
            well  : An object representing the well to be simulated.
            pinit (float, optional): Initial reservoir pressure. Defaults to None.

        Returns:
            SteadyState: Returns self to allow for method chaining or deferred execution.
        
        """
        self.well = well
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
        """Solves for the pressure values at steady state."""
        result = Result(self.correct(times),nodes)

        drop = self._term*(np.log(self._radius/self.well._radius)-3/4+self.well._skin)

        result._press = self._pinit-drop

        return result
    
    
    
    
