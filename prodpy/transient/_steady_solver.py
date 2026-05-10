import numpy as np

from ._base_solver import BaseSolver

from ._result import Result

class SteadySolver(BaseSolver):
    """
    Steady state solution of the diffusivity equation in radial coordinates using 
    the line source solution based on the exponential integral.

    Inherits from:
        PorousMedia: Provides radial porous media properties.
        BaseSolver: Handles base solver configurations and behaviors.

    """

    def __init__(self,*args,**kwargs):
        """Initializes steady state constant rate radial solution of
        diffusivity equation.

        Args:
            size (float tuple): porous media size for PorousMedia.
            **kwargs: Keyword arguments for BaseSolver.

        """
        super().__init__(*args,**kwargs)

    def __call__(self,well,pinit:float=None):
        """
        Prepares the Steady-State solver instance for execution by configuring 
        the well and optional shape and initial pressure.

        Args:
            well  : An object representing the well to be simulated.
            pinit (float, optional): Initial reservoir pressure. Defaults to None.

        Returns:
            SteadySolver: Returns self to allow for method chaining or deferred execution.
        
        """
        self.well = well
        self.term = None

        self.pinit = pinit

        return self

    def solve(self,times,nodes):
        """Solves for the pressure values at steady state."""
        result = Result(self.correct(times),nodes)

        drop = self._term*(np.log(self._radius/self.well._radius)-3/4+self.well._skin)

        result._press = self._pinit-drop

        return result
    
    
    
    
