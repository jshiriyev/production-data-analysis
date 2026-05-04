import numpy

from ._base import GridBase

class Grids(GridBase):
    """Represents a structured reservoir grid with additional table attribute."""

    def __init__(self,xdelta:numpy.ndarray,ydelta:numpy.ndarray,zdelta:numpy.ndarray,depths:numpy.ndarray,table:numpy.ndarray):
        """
        Initialize the grid with spatial discretization and associated table data.

        Parameters:
        xdelta (np.ndarray): Grid cell sizes in the x-direction (feet).
        ydelta (np.ndarray): Grid cell sizes in the y-direction (feet).
        zdelta (np.ndarray): Grid cell sizes in the z-direction (feet).

        depths (np.ndarray): Grid depths (feet).

        table  (np.ndarray): Data table mapping the neighbors of each grid, integers.

        """
        super().__init__(xdelta,ydelta,zdelta,depths)

        self.xarea  = None # Placeholder for volume calculations
        self.yarea  = None # Placeholder for volume calculations
        self.zarea  = None # Placeholder for volume calculations

        self.volume = None # Placeholder for volume calculations

        self.table  = table

    @property
    def delta(self):
        """Returns the cell sizes in x, y, and z direction, shape = (nums,3)."""
        return numpy.column_stack(
            (self.xdelta,self.ydelta,self.zdelta)
            )

    @property
    def xarea(self):
        return self._xarea*10.7639

    @xarea.setter
    def xarea(self,value):
        self._xarea = self._ydelta*self._zdelta

    @property
    def yarea(self):
        return self._yarea*10.7639

    @yarea.setter
    def yarea(self,value):
        self._yarea = self._zdelta*self._xdelta

    @property
    def zarea(self):
        return self._zarea*10.7639

    @zarea.setter
    def zarea(self,value):
        self._zarea = self._xdelta*self._ydelta
    
    @property
    def volume(self):
        """Returns the volume of grids in field units."""
        return self._volume*35.3147

    @volume.setter
    def volume(self,value):
        """Calculates the volume of each cell."""
        self._volume = numpy.prod((self._xdelta,self._ydelta,self._zdelta),axis=0)

    @property
    def nums(self):
        """Returns total number of grids."""
        return self.table.shape[0]

    @property
    def dims(self):
        """Returns the flow dimensions."""
        return int(self.table.shape[1]/2)

    @property
    def index(self):
        """Returns the indices of grids."""
        return numpy.arange(self.nums)

    @property
    def xmin(self):
        """Returns x-minimum boundary indices."""
        return self.index[self._xmin]

    @property
    def _xmin(self):
        """Returns x-minimum boundary boolean."""
        return self.index==self.table[:,0]

    @property
    def xpos(self):
        """Returns x-positive neighbor indices."""
        return self.index[self._xpos]

    @property
    def _xpos(self):
        """Returns x-positive neighbor boolean."""
        return self.index!=self.table[:,0]

    @property
    def xneg(self):
        """Returns x-negative neighbor indices."""
        return self.index[self._xneg]

    @property
    def _xneg(self):
        """Returns x-negative neighbor boolean."""
        return self.index!=self.table[:,1]

    @property
    def xmax(self):
        """Returns x-maximum boundary indices."""
        return self.index[self._xmax]

    @property
    def _xmax(self):
        """Returns x-maximum boundary boolean."""
        return self.index==self.table[:,1]

    @property
    def ymin(self):
        """Returns y-minimum boundary indices."""
        return self.index[self._ymin]

    @property
    def _ymin(self):
        """Returns y-minimum boundary boolean."""
        return self.index==self.table[:,2] if self.dims>1 else numpy.full(self.nums,True)

    @property
    def ypos(self):
        """Returns y-positive neighbor indices."""
        return self.index[self._ypos]

    @property
    def _ypos(self):
        """Returns y-positive neighbor boolean."""
        return self.index!=self.table[:,2] if self.dims>1 else numpy.full(self.nums,False)

    @property
    def yneg(self):
        """Returns y-negative neighbor indices."""
        return self.index[self._yneg]

    @property
    def _yneg(self):
        """Returns y-negative neighbor boolean."""
        return self.index!=self.table[:,3] if self.dims>1 else numpy.full(self.nums,False)

    @property
    def ymax(self):
        """Returns y-maximum boundary indices."""
        return self.index[self._ymax]

    @property
    def _ymax(self):
        """Returns y-maximum boundary boolean."""
        return self.index==self.table[:,3] if self.dims>1 else numpy.full(self.nums,True)

    @property
    def zmin(self):
        """Returns z-minimum boundary indices."""
        return self.index[self._zmin]

    @property
    def _zmin(self):
        """Returns z-minimum boundary boolean."""
        return self.index==self.table[:,4] if self.dims>2 else numpy.full(self.nums,True)

    @property
    def zpos(self):
        """Returns z-positive neighbor indices."""
        return self.index[self._zpos]

    @property
    def _zpos(self):
        """Returns z-positive neighbor boolean."""
        return self.index!=self.table[:,4] if self.dims>2 else numpy.full(self.nums,False)

    @property
    def zneg(self):
        """Returns z-negative neighbor indices."""
        return self.index[self._zneg]

    @property
    def _zneg(self):
        """Returns z-negative neighbor boolean."""
        return self.index!=self.table[:,5] if self.dims>2 else numpy.full(self.nums,False)

    @property
    def zmax(self):
        """Returns z-maximum boundary indices."""
        return self.index[self._zmax]

    @property
    def _zmax(self):
        """Returns z-maximum boundary boolean."""
        return self.index==self.table[:,5] if self.dims>2 else numpy.full(self.nums,True)