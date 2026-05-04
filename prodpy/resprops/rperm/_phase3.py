import os, sys

def phase3(method="stones_I",**kwargs):
	"""models = ["Stones I","Aziz Settari","Stones II","Hustad Holt"]"""
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
	method_instance = mclass(**kwargs)

	return method_instance


    