from numpy import ndarray

class Vector():

    def __init__(self,A:ndarray,X:ndarray,Y:ndarray,Z:ndarray,W:list,B:list):
        """
        Initializes vector interface with the following items in SI units:
        
        A   : Block storage term that includes compressibility and time step

        X   : Face transmissibility in x-direction
        Y   : Face transmissibility in y-direction
        Z   : Face transmissibility in z-direction

        W   : Well block productivity values
        B   : Edge block transmissibility values
        """

        self._A = A # block accumulation

        self._X = X # x-transmissibility
        self._Y = Y # y-transmissibility
        self._Z = Z # z-transmissibility

        self._W = W # list of well productivity
        self._B = B # list of edge transmissibility

    @property
    def A(self):
        """Converting from SI Units to Oil Field Units."""
        return self._A*(3.28084**3)*(24*60*60)*6894.76

    @property
    def X(self):
        """Converting from SI Units to Oil Field Units."""
        return self._X*(3.28084**3)*(24*60*60)*6894.76

    @property
    def Y(self):
        """Converting from SI Units to Oil Field Units."""
        return self._Y*(3.28084**3)*(24*60*60)*6894.76

    @property
    def Z(self):
        """Converting from SI Units to Oil Field Units."""
        return self._Z*(3.28084**3)*(24*60*60)*6894.76

    @property
    def W(self):
        """Converting from SI Units to Oil Field Units."""
        return [well._prod*(3.28084**3)*(24*60*60)*6894.76 for well in self._W]
    
    @property
    def B(self):
        """Converting from SI Units to Oil Field Units."""
        return [edge._prod*(3.28084**3)*(24*60*60)*6894.76 for edge in self._B]

