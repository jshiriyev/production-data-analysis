import os, sys

import numpy as np

def zfactor(critical_params:tuple,pressures:np.ndarray,temperature:float,derivative:bool=False,method:str="direct_method"):
	"""
	Dynamically calculates z-factor based on the specified method.

	Parameters:
	    critical_params (tuple): tuple of (pcrit in psi, tcrit in Rankine)

	    pressures (np.ndarray): Array of pressure values (psi) to calculate z-factor for.
	    temperature (float): The temperature value (Rankine) at which to calculate z-factor.
	    
	    method_name (str): The name of the method to use:
	    	(e.g., 'direct_method', 'dranchuk_abu_kassem', 'dranchuk_purvis_robinson', 'hall_yarborough').
	    
	Returns:
	    np.ndarray: Z-factor, and if derivative is True, Z-prime values calculated for each pressure.
	    
	"""
	sys.path.append(os.path.dirname(__file__))

	method_parts = method.split('_')

	method_class = method_parts[0].capitalize()+''.join(word.capitalize() for word in method_parts[1:])

	# Import the correct method class dynamically
	try:
		module = __import__(method)
		mclass = getattr(module,method_class)
	except (ImportError, AttributeError):
		raise ValueError(f"Method '{method}' not found or invalid.")

	# Create an instance of the class and calculate z-factor
	method_instance = mclass(critical_params,temperature)

	return method_instance(pressures,derivative)