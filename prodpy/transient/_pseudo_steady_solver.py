import logging

import numpy as np

from ._solver_object import SolverObj
from ._radial_pormed import RadPorMed

from ._result import Result

@dataclass(frozen=True)
class Boundary:

    #shape factor, {C_A} value
    factor: float

    # PSS is exact for higher values
    time_pss_accurate: float = None
    # PSS gives less than 1% error for higher values
    time_pss_error_prone: float = None
    # Use infinite system solution with less than 1 % Error for lesser values
    time_infinite: float = None

class PseudoSteadyState(RadPorMed,SolverObj):
    """Pseudo Steady State solution based on shape factors.

    The solver is also applicable when there are two slightly compressible fluids
    where the second one is at irreducible saturation, not mobile.
    
    Inherits from:
        RadPorMed: Provides radial porous media properties.
        SolverObj: Handles base solver configurations and behaviors.

    """
    GAMMA = np.exp(0.5772)

    GEOMETRY = {
        "circle": Boundary(31.62,0.1,0.06,0.1),
        "triangle": Boundary(27.6,0.2,0.07,0.09),
        "square": Boundary(30.8828,0.1,0.05,0.09),
        "hexagon": Boundary(31.6,0.1,0.06,0.1),
        }

    def __init__(self,size:tuple,**kwargs):
        """
        Initializes the Pseudo-State solver by invoking the initializers of 
        RadPorMed and SolverObj.

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

    def __call__(self,well,shape:str="circle",pinit:float=None):
        """
        Prepares the Pseudo-State solver instance for execution by configuring 
        the well and optional shape and initial pressure.

        Args:
            well  : An object representing the well to be simulated.
            shape : The shape of the reservoir.
            pinit (float, optional): Initial reservoir pressure. Defaults to None.

        Returns:
            PseudoSteadyState: Returns self to allow for method chaining or deferred execution.
        
        """

        self.well  = well
        self.shape = shape
        self.tmin  = None
        self.term  = None

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
    def shape(self):
        """Getter for the reservoir boundary shape."""
        return self._shape

    @shape.setter
    def shape(self,value):
        """Setter for the reservoir boundary shape."""
        self._shape = value

    @property
    def bound(self):
        """Getter for the reservoir shape parameters."""
        return self.GEOMETRY[self.shape]

    @property
    def tmin(self):
        """Getter for the minimum time where PSS solution is applicable."""
        return self._tmin/(24*60*60)

    @tmin.setter
    def tmin(self,value):
        """Setter for the minimum time where PSS solution is applicable."""
        self._tmin = self.dim2t(self.bound.time_pss_accurate)*(24*60*60)

    def t2dim(self,values:np.ndarray):
        """Converts time values to dimensionless time values."""
        return (self._hdiff/self._surface)*np.asarray(values)*(24*60*60)

    def dim2t(self,values:np.ndarray):
        """Converts dimensionless time values to time values."""
        return (self._surface/self._hdiff)*np.asarray(values)/(24*60*60)

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
        """Solves for the pressure values at pseudo-steady state."""
        result = Result(self.correct(times),nodes)

        inner = (4*self._surface)/(self.GAMMA*self.bound.factor*self.well._radius**2)
        drop1 = self._term*(1/2*np.log(inner)+self.well._skin)
        drop2 = (self.well._cond*self.fluid._fvf)/(self._vpore*self._tcomp)*result._times

        result._press = self._pinit-drop1-drop2

        return result

    def correct(self,values):
        """It sets the time values to np.nan if it is outside of the solver limit."""
        boundary = values>=self.tmin

        if np.any(~boundary):
            logging.warning("Not all times satisfy the early time limits!")

        return np.where(boundary,values,np.nan)
