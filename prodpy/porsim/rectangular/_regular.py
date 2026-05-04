import numpy

from ._delta import GridDelta

class GridRegular(GridDelta):
	"""Represents a regular, three-dimensional rectangular cuboid with uniform grid spacing."""

	def __init__(self,size:tuple[float,float,float],nums:tuple[int,int,int],depths:float|numpy.ndarray=1000.,dims:int=None):
		"""
		Initializes a structured reservoir grid with uniform cell spacing.

		Parameters
		----------
        size (tuple) : Reservoir dimensions (x-length, y-width, z-height) in feet.
        nums (tuple) : Number of grid cells along each axis (xnums, ynums, znums).
		
		depths (np.ndarray): Grid depths (feet).

        dims (int)	: (optional) Flow dimension (1, 2, or 3). Defaults to None.

		"""
		xdelta = numpy.full(nums[0],size[0]/nums[0])
		ydelta = numpy.full(nums[1],size[1]/nums[1])
		zdelta = numpy.full(nums[2],size[2]/nums[2])

		super().__init__(xdelta,ydelta,zdelta,depths,dims)

if __name__ == "__main__":

	grid = GridRegular((1000,2000,30),(3,4,1))

	print(grid.dims)

	print(grid.xmin)

