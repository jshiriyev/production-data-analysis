import math

class Pipe():
    """A class representing a cylindrical pipe, providing its geometric properties.
    
    Attributes:
        di (float) : The inner diameter of the pipe, inch.
        ll (float) : The length of the pipe (default is 1.0), feet.
        rr (float) : Relative roughness of the pipe (dimensionless).
    
    Properties:
        radius (float)  : Returns the inner radius of the pipe.
        circ (float)    : Returns the inner circumference of the pipe.
        surface (float) : Returns the inner surface area of the pipe.
        csa (float)     : Returns the inner cross-sectional area of the pipe.
        volume (float)  : Returns the inner volume of the pipe (assuming a solid cylinder).

    """
    def __init__(self,di:float,ll:float=1.,rr:float=None):
        """Initializes a Pipe object with diameter, length, and relative roughness."""
        self.di = di
        self.ll = ll
        self.rr = rr

    @property
    def diameter(self):
        """Getter for the inner diameter."""
        return self.di

    @property
    def diam(self):
        """Getter for the inner diameter."""
        return self.di

    @property
    def di(self):
        """Getter for the inner diameter."""
        return self._di/0.0254

    @di.setter
    def di(self,value):
        """Setter for the inner diameter."""
        self._di = value*0.0254

    @property
    def length(self):
        """Getter for the pipe length."""
        return self.ll
    
    @property
    def ll(self):
        """Getter for the pipe length."""
        return self._ll/0.3048

    @ll.setter
    def ll(self,value):
        """Setter for the pipe length."""
        self._ll = value*0.3048

    @property
    def epd(self):
        """Getter for the relative roughness."""
        return self.rr
    
    @property
    def rr(self):
        """Getter for the relative roughness."""
        return self._rr

    @rr.setter
    def rr(self,value):
        """Setter for the relative roughness."""
        self._rr = value

    @property
    def radius(self):
        """Getter for the pipe radius, inches."""
        if not hasattr(self,"_radius"):
            self.radius = None

        return self._radius/0.0254

    @radius.setter
    def radius(self,value):
        """Returns the radius of the pipe."""
        self._radius = self._di / 2

    @property
    def circ(self):
        """Getter for the pipe circumference, ft."""
        if not hasattr(self,"_circ"):
            self.circ = None

        return self._circ/0.3048

    @circ.setter
    def circ(self,value):
        """Returns the outer circumference of the pipe."""
        self._circ = math.pi * self._di

    @property
    def surface(self):
        """Getter for the pipe surface area, ft2."""
        if not hasattr(self,"_surface"):
            self.surface = None

        return self._surface/0.3048**2

    @surface.setter
    def surface(self,value):
        """Returns the external surface area of the pipe (excluding ends)."""
        self._surface = self._circ * self._ll

    @property
    def csa(self):
        """Getter for the pipe cross-sectional-area, ft2."""
        if not hasattr(self,"_csa"):
            self.csa = None

        return self._csa/0.3048**2

    @csa.setter
    def csa(self,value):
       """Getter for the cross-sectional area of the pipe."""
       self._csa = math.pi*(self._di**2)/4

    @property
    def volume(self):
        """Getter for the pipe volume, ft3."""
        if not hasattr(self,"_volume"):
            self.volume = None

        return self._volume/0.3048**3

    @volume.setter
    def volume(self,value):
        """Returns the volume of the pipe (assuming a solid cylinder)."""
        self._volume = self._csa * self._ll

if __name__ == "__main__":

    pipe = Pipe(1,5)

    pipe.csa

    print(pipe.length)

    print(pipe._csa)