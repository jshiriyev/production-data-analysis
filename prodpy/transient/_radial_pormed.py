import numpy as np

class RadPorMed():
    """A reservoir class for radial flow calculations."""

    _FT_TO_METER = 0.3048

    def __init__(self,size:tuple,/):
        """Initialize the base reservoir class for analytical calculations.

        Arguments:
        ---------
        size (float tuple): tuple of (radius, height) in feet. If only radius is provided,
            height defaults to 1 ft.

        """
        self.size = size

    @property
    def size(self):
        """Getter for the reservoir size."""
        return (self.radius, self.height)

    @size.setter
    def size(self,value):
        """Setter for the reservoir size."""
        if not isinstance(value, tuple):
            raise TypeError("Size must be a tuple: (radius,) or (radius, height)")
        if len(value) == 0 or len(value) > 2:
            raise ValueError("Size tuple must have 1 or 2 elements")

        self.radius = value[0]
        self.height = value[1] if len(value) == 2 else 1.

        self._size = (self._radius,self._height)

    @property
    def radius(self):
        """Getter for the reservoir radius."""
        return self._radius/self._FT_TO_METER

    @radius.setter
    def radius(self,value):
        """Setter for the reservoir radius."""
        self._radius = value*self._FT_TO_METER

    @property
    def height(self):
        """Getter for the reservoir height."""
        return self._height/self._FT_TO_METER

    @height.setter
    def height(self,value):
        """Setter for the reservoir height."""
        self._height = value*self._FT_TO_METER

    @property
    def surface(self):
        """Getter for the reservoir surface area perpendicular to flow."""
        if not hasattr(self,"_surface"):
            self.surface = None

        return self._surface/(self._FT_TO_METER**2)

    @surface.setter
    def surface(self,value):
        """Setter for the reservoir surface area perpendicular to flow."""
        self._surface = np.pi*self._radius**2

    @property
    def volume(self):
        """Getter for the reservoir volume."""
        if not hasattr(self,"_volume"):
            self.volume = None

        return self._volume/(self._FT_TO_METER**3)

    @volume.setter
    def volume(self,value):
        """Setter for the reservoir volume."""
        self._volume = np.pi*self._radius**2*self._height

if __name__ == "__main__":

    res = RadPorMed((1000,20))

    print(res.size)
    print(res._size)

    print(res.radius)

    print(res.surface)

    print(res.volume)