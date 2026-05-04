import numpy

def viscosity(spgr,press,temp,*args,method="carr_kobayashi_burrows",**kwags):
	"""
	Dynamically calculates viscosity based on the specified method.

	Parameters:
	    spgr    : specific gravity of the gas at standard conditions.

	    press 	: pressure

        temp    : temperature value (Rankine) which will be used to calculate
                  viscosities when the class is called.
	    
	Returns:
	    numpy.ndarray: viscosity values calculated for each pressure.
	    
	"""

	method_parts = method.split('_')

	method_class = method_parts[0].capitalize()+''.join(word.capitalize() for word in method_parts[1:])

	# Import the correct method class dynamically
	try:

		module = __import__(f'respy.fluid_props.phaseg.viscosity_models',fromlist=[method_class])

		MethodClass = getattr(module,method_class)

	except (ImportError, AttributeError):
		raise ValueError(f"Method '{method}' not found or invalid.")

	# Create an instance of the class and calculate z-factor
	method_instance = MethodClass(spgr,temp,*args)

	return method_instance(pressures)