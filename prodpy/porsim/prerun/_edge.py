from ._const import Constraint

class EdgeBound(Constraint):
    """
    Represents the edges of reservoir in the simulator.
    """

    def __init__(self,face:str,**kwargs):
        """
        Parameters
        ----------
        face    : boundary: xmin, xmax, ymin, ymax, zmin, or zmax

        start   : start time for implementing the boundary condition, days
        stop    : stop time for implementing the boundary condition, days

        **kwargs : dict
        Specifies the well constraint. Only one of the following should be provided:
        - 'press' : Constant bottom-hole pressure (psi)
        - 'lrate' : Constant liquid rate, (bbl/day)
        - 'orate' : Constant oil rate, (bbl/day)
        - 'wrate' : Constant water rate, (bbl/day)
        - 'grate' : Constant gas rate, (ft3/day)
        """
        self.face  = face

        super().__init__(**kwargs)

    @property
    def face(self):
        """Getter for edge face in grids."""
        return self._face

    @face.setter
    def face(self,value:str):
        """Getter for edge face in grids."""
        self._face = value

    @property
    def axis(self):
        return self._face[0]    

if __name__ == "__main__":

    bcond = EdgeBound("xmin",press=500)

    print(bcond.cond)
    print(bcond.face)
    print(bcond.axis)
    print(bcond.sort)