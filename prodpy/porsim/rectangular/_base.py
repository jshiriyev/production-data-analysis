import numpy

class GridBase():
	"""Represents a base class for gridding with cell dimensions stored in meters."""

	def __init__(self,xdelta:numpy.ndarray,ydelta:numpy.ndarray,zdelta:numpy.ndarray,depths:float|numpy.ndarray=1000.):
		"""
		Initialize grid cell dimensions in feet.

		Parameters
		----------
		xdelta (float or array-like): Grid cell size in the x-direction (feet).
		ydelta (float or array-like): Grid cell size in the y-direction (feet).
		zdelta (float or array-like): Grid cell size in the z-direction (feet).

		depths (float or array-like): Grid depths (feet).
		"""
		self.xdelta = xdelta # ft
		self.ydelta = ydelta # ft
		self.zdelta = zdelta # ft
		self.depths = depths # ft
		
	@property
	def xdelta(self):
		"""Returns x-direction grid cell size in feet."""
		return self._xdelta/0.3048

	@xdelta.setter
	def xdelta(self,value):
		"""Sets x-direction grid cell size after converting from feet to meters."""
		self._xdelta = numpy.ravel(value).astype(float)*0.3048

	@property
	def ydelta(self):
		"""Returns y-direction grid cell size in feet."""
		return self._ydelta/0.3048

	@ydelta.setter
	def ydelta(self,value):
		"""Sets y-direction grid cell size after converting from feet to meters."""
		self._ydelta = numpy.ravel(value).astype(float)*0.3048

	@property
	def zdelta(self):
		"""Returns z-direction grid cell size in feet."""
		return self._zdelta/0.3048

	@zdelta.setter
	def zdelta(self,value):
		"""Sets z-direction grid cell size after converting from feet to meters."""
		self._zdelta = numpy.ravel(value).astype(float)*0.3048

	@property
	def depths(self):
		"""Returns the depths of grids in feet."""
		return self._depths/0.3048

	@depths.setter
	def depths(self,value):
		"""Setter for the depths of grids in meters."""
		self._depths = numpy.ravel(value).astype(float)*0.3048

if __name__ == "__main__":

	gb = GridBase([2,2,3],[3,4],5)

	print(gb.xdelta)
	print(gb.ydelta)
	print(gb.zdelta)

	print(gb._xdelta)
	print(gb._ydelta)
	print(gb._zdelta)