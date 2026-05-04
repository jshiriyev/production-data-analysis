from ._const import Constraint

class WellBound(Constraint):
    """
    Represents a well object used in a reservoir simulator.
    """

    def __init__(self,index:tuple,*,axis:str="z",radius:float=0.5,skin:float=0,**kwargs):
        """
        Parameters
        ----------
        index   : tuple[int, ...]
            All index indices containing the well
        axis    : str, optional
            Well orientation ('z' for vertical, 'x' or 'y' for horizontal wells), default is 'z'
        radius  : float, optional
            Well radius in feet, default is 0.5
        skin    : float, optional
            Skin factor of the well (dimensionless), default is 0
        start   : float, optional
            Start time for implementing the well condition (days), default is 0 meaning the start of simulation
        stop    : float, optional
            Stop time for implementing the well condition (days), default is None meaning the end of simulation

        **kwargs : dict
        Specifies the well constraint. Only one of the following should be provided:
        - 'press' : Constant bottom-hole pressure (psi)
        - 'lrate' : Constant liquid rate, (bbl/day)
        - 'orate' : Constant oil rate, (bbl/day)
        - 'wrate' : Constant water rate, (bbl/day)
        - 'grate' : Constant gas rate, (ft3/day)
        
        """
        self.index  = index

        self.axis   = axis
        self.radius = radius
        self.skin   = skin

        super().__init__(**kwargs)
        
    @property
    def axis(self):
        """Getter for well axis in grids."""
        return {0:"x",1:"y",2:"z"}[self._axis]
    
    @axis.setter
    def axis(self,value):
        """Setter for well axis in grids."""
        self._axis = {"x":0,"y":1,"z":2}[value]

    @property
    def radius(self):
        """Getter for well radius."""
        return self._radius/0.3048

    @radius.setter
    def radius(self,value):
        """Setter for well radius."""
        self._radius = value*0.3048

    @property
    def skin(self):
        """Getter for well skin."""
        return self._skin
    
    @skin.setter
    def skin(self,value):
        """Setter for well skin."""
        self._skin = value

if __name__ == "__main__":

    well = WellBound((3,),axis="z",radius=0.5,wrate=500,start=3)

    print(well.axis)
    print(well._axis)
    print(well.sort)
    print(well.cond)
    print(well._cond)

    print(well.start)
    print(well._start)