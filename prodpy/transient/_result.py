import numpy as np

class Result():
    """
    A class to store and process transient flow solutions in porous media.

    Attributes:
    -----------
    times : np.ndarray
        Time values (in days) converted to seconds.
    nodes : np.ndarray
        Spatial node positions (in feet) converted to meters.
    press : np.ndarray
        Pressure values (in psi) converted to Pascals.
        
    """

    def __init__(self,times,nodes):
        """Initializes the Result class with time and node values.

        Parameters:
        -----------
        times : np.ndarray
            Time values in days.
        nodes : np.ndarray
            Node positions in feets.

        """
        self.times = times
        self.nodes = nodes

    @property
    def times(self):
        """Get time values converted from seconds to days."""
        return self._times/(24*60*60)

    @times.setter
    def times(self,values:np.ndarray):
        """Set time values, ensuring they are stored as a row vector in seconds."""
        self._times = np.ravel(values).astype(float).reshape((1,-1))*(24*60*60)

    @property
    def nodes(self):
        """Get node positions converted from meters to feet."""
        return self._nodes/0.3048

    @nodes.setter
    def nodes(self,values:np.ndarray):
        """Set node positions, ensuring they are stored as a column vector in meters."""
        self._nodes = np.ravel(values).astype(float).reshape((-1,1))*0.3048

    @property
    def press(self):
        """Get pressure values converted from Pascals to psi."""
        return self._press/6894.76

    @press.setter
    def press(self,values):
        """Set pressure values, converting from psi to Pascals."""
        self._press = values*6894.76