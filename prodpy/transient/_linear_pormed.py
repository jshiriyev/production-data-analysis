class LinPorMed():
    """A reservoir class for one-dimensional linear flow calculations."""

    _FT_TO_METER = 0.3048

    def __init__(self,size:tuple,/):
        """Initialize the base reservoir class for analytical calculations.

        Arguments:
        ----------
        size (float tuple): tuple of (length, width, height) in feet. If only length is provided,
            width and height defaults to 1 ft.

        """
        self.size = size

    @property
    def size(self):
        """Getter for the reservoir size."""
        return (self.length, self.width, self.height)

    @size.setter
    def size(self,value):
        """Setter for the reservoir size."""
        if not isinstance(value, tuple):
            raise TypeError("Size must be a tuple: (length,) or (length, width) or (length, width, height)")
        if len(value) == 0 or len(value) > 3:
            raise ValueError("Size tuple must have 1, 2 or 3 elements")

        self.length = value[0]
        self.width  = value[1] if len(value) > 1 else 1.
        self.height = value[2] if len(value) > 2 else 1.

        self._size = (self._length, self._width, self._height)

    @property
    def length(self):
        """Getter for the reservoir length."""
        return self._length/self._FT_TO_METER

    @length.setter
    def length(self,value):
        """Setter for the reservoir length."""
        self._length = value*self._FT_TO_METER

    @property
    def width(self):
        """Getter for the reservoir width."""
        return self._width/self._FT_TO_METER

    @width.setter
    def width(self,value):
        """Setter for the reservoir width."""
        self._width = value*self._FT_TO_METER

    @property
    def height(self):
        """Getter for the reservoir height."""
        return self._height/self._FT_TO_METER

    @height.setter
    def height(self,value):
        """Setter for the reservoir height."""
        self._height = value*self._FT_TO_METER

    @property
    def area(self):
        """Getter for the reservoir cross-sectional area parallel to flow."""
        if not hasattr(self,"_area"):
            self.area = None

        return self._area/(self._FT_TO_METER**2)

    @area.setter
    def area(self,value):
        """Setter for the reservoir cross-sectional area whose normal is parallel to flow."""
        self._area = self._height*self._width

    @property
    def surface(self):
        """Getter for the reservoir surface area perpendicular to flow."""
        if not hasattr(self,"_surface"):
            self.surface = None

        return self._surface/(self._FT_TO_METER**2)

    @surface.setter
    def surface(self,value):
        """Setter for the reservoir surface area whose normal is perpendicular to flow."""
        self._surface = self._length*self._width

    @property
    def volume(self):
        """Getter for the reservoir volume."""
        if not hasattr(self,"_volume"):
            self.volume = None

        return self._volume/(self._FT_TO_METER**3)

    @volume.setter
    def volume(self,value):
        """Setter for the reservoir volume."""
        self._volume = self._length*self._height*self._width

if __name__ == "__main__":

    res = LinPorMed((1000,))

    print(res.size)
    print(res._size)
    print(res.length)

    print(res.area)
    print(res.volume)