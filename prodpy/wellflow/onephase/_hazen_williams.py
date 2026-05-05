from ._pressure_drop import PressureDrop

class HazenWilliams(PressureDrop):

	def __init__(self,*args,**kwargs):
		"""
		Initializes the HazenWilliams class by inheriting from PressureDrop.
		
		Args:
			*args: Positional arguments passed to the parent class.
			**kwargs: Keyword arguments passed to the parent class.

		"""
		super().__init__(*args,**kwargs)

	def get(self,flow_rate,C:float=120.):
		"""Returns the head loss due to friction."""
		return (10.67 * self.pipe.length * flow_rate**1.852) / (C**1.852 * self.pipe.diam**4.87)