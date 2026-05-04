import numpy as np

from .circle_segment import CircleSegment

class CircularConduit:
    """
    Represent the segments cut from a circle by a horizontal bound.

    A segment is the region bounded by:
    - a circular arc, and
    - the bound connecting the arc endpoints.

    There are two segments in the conduit, lower and upper.

    The height `h` is measured vertically from the bound down to the lowest
    point of the circle in the lower segment. Therefore:

    - `h = 0`   -> Lower segment has zero area, and upper segment is full circle.
    - `h = R`   -> Lower and upper segments are semicircle.
    - `h = D`   -> Lower area is full circle, and upper segment has zero area.

    Parameters
    ----------
    D : float
        Circle diameter. Must be strictly positive.
    h : float
        CircleSegment height measured from the bound to the lowest point of the circle. Must satisfy ``0 <= h <= D``.

    Notes
    -----
    Let ``R = D / 2``. Define ``theta`` as the central angle of the segment
    in radians for the lower segment, and the upper angle is its complement:

    .. math::
        \\theta = 2\\cos^{-1}\\left(\\frac{R - h}{R}\\right)

    Then:

    Boundary length
        .. math::
            L_{bound} = 2\\sqrt{2Rh - h^2}

    Arc length
        .. math::
            L_{arc} = \\theta R

    CircleSegment area
        .. math::
            A = R^2 /2 \\theta - (R - h)\\sqrt{2Rh - h^2}

    Examples
    --------
    >>> seg = CircularConduit(10, 5)
    >>> seg.R
    5.0
    >>> seg.area(upper=False)
    39.269908169872416
    >>> seg.arc(upper=False)
    15.707963267948966
    >>> seg.chord()
    10.0
    """

    def __init__(self, D:float, h:float):
        """
        Initialize a circular conduit partitioned by a horizontal bound.

        Parameters
        ----------
        D : float
            Circle diameter.
        h : float
            CircleSegment height measured from the bound to the bottom of the circle.

        Raises
        ------
        ValueError
            If `D <= 0` or if `h` is outside the interval [0, D].

        """
        self.D = float(D)
        self.h = float(h)

        self._diameter_validation()
        self._height_validation()

    def _diameter_validation(self) -> None:
        """Validate that the diameter is strictly positive."""
        CircleSegment._validate_positive_finite(self.D, "Diameter D")

        if np.isclose(self.D, 0.0):
            raise ValueError("Diameter D must satisfy 0 > D.")

    def _height_validation(self) -> None:
        """Validate that the segment height lies within the circle."""
        CircleSegment._validate_positive_finite(self.h, "Height h")

        if self.h > self.D:
            raise ValueError("Height h must satisfy 0 <= h <= D.")
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"  - The inner diameter of the conduit :{self.D}\n"
            f"  - The height of lower phase :{self.h:.3f}"
        )
    
    @property
    def R(self) -> float:
        """Returns float : Circle radius."""
        return self.D / 2.0
    
    @property
    def P(self) -> float:
        """Returns float: Circle circumference."""
        return np.pi*self.D
    
    @property
    def A(self) -> float:
        """Returns float : Circle area."""
        return np.pi * self.D**2 / 4
    
    @property
    def only_upper_segment(self) -> bool:
        return np.isclose(self.h, 0.0)
    
    @property
    def only_lower_segment(self) -> bool:
        return np.isclose(self.h, self.D)
    
    @property
    def no_partition(self) -> bool:
        return self.only_upper_segment or self.only_lower_segment
    
    @property
    def equal_partition(self) -> bool:
        return np.isclose(self.h, self.R)
    
    @property
    def upper(self) -> CircleSegment:
        """Return the upper segment of the conduit as a CircleSegment object."""
        return CircleSegment(
            self.chord(),
            self.arc(upper=True),
            self.area(upper=True),
            self.major(upper=True),
            )
    
    @property
    def lower(self) -> CircleSegment:
        """Return the lower segment of the conduit as a CircleSegment object."""
        return CircleSegment(
            self.chord(),
            self.arc(upper=False),
            self.area(upper=False),
            self.major(upper=False),
            )
    
    def __call__(self, added_area: float, **kwargs) -> None:

        CircleSegment._validate_positive_finite(added_area, "Added area")

        new_area = self._height2area(self.h) + added_area

        if new_area >= self.A:
            raise ValueError("The cross-section is filled.")

        self.h = self._area2height(new_area, **kwargs)
    
    def _area2height(self, area: float, tol: float = 1e-10, max_iter: int = 100) -> float:
        """Returns the height of the lower segment based on its area using the iterative bisection method."""
        lower, upper = 0.0, self.D

        for _ in range(max_iter):
            mid = 0.5 * (lower + upper)
            
            area_new = self._height2area(mid)

            if abs(area_new - area) < tol:
                return mid

            if area_new < area:
                lower = mid
            else:
                upper = mid

        return 0.5 * (lower + upper)
        
    def theta(self, upper:bool = True) -> float:
        """Returns the central angle of the requested segment in radians."""
        if self.only_upper_segment:
            return 2 * np.pi if upper else 0.0
        if self.only_lower_segment:
            return 0.0 if upper else 2 * np.pi
        
        lower_theta = self._height2theta(self.h)
        
        return 2*np.pi-lower_theta if upper else lower_theta
    
    def _height2theta(self, height:float) -> float:
        """Convert the lower segment height to central angle (theta)."""
        
        cos_arg = (self.R - height) / self.R
        cos_arg = np.clip(cos_arg, -1.0, 1.0)

        return 2 * np.arccos(cos_arg)
    
    def chord(self) -> float:
        """Length of the chord, internal boundary of segments."""
        if self.no_partition:
            return 0.0

        return self._height2chord(self.h)

    def _height2chord(self, height:float) -> float:
        """Returns the chord length based on the height."""
        return 2 * np.sqrt(self.D * height - height**2)

    def arc(self, upper:bool = True) -> float:
        """
        Returns the arc length of the defined segment.

        Special cases:
        - ``h = 0`` -> arc length = 0 for the lower segment, and 2πR for the upper segment
        - ``h = R`` -> arc length = πR for both segments
        - ``h = D`` -> arc length = 2πR for lower segment, and 0 for the upper segment
        """
        if self.only_upper_segment:
            return self.P if upper else 0.0
        if self.only_lower_segment:
            return 0.0 if upper else self.P
        
        lower_arc = self._height2arc(self.h)

        return self.P - lower_arc if upper else lower_arc
    
    def _height2arc(self, height:float) -> float:
        """Convert segment height to the arc length of the lower segment."""
        return self._height2theta(height) * self.R

    def area(self, upper:bool = True) -> float:
        """
        Returns the area of the given segment based on the height.

        Special cases:
        - ``h = 0`` -> area = 0 for the lower segment, and π R² for the upper segment
        - ``h = R`` -> area = (1/2) π R² for both segments
        - ``h = D`` -> area = π R² for the lower segment, and 0 for the upper segment
        """
        if self.only_upper_segment:
            return self.A if upper else 0.0
        if self.only_lower_segment:
            return 0.0 if upper else self.A
        
        lower_area = self._height2area(self.h)

        return self.A - lower_area if upper else lower_area
    
    def _height2area(self, height:float) -> float:
        """Returns the area of the lower segment based on its height."""
        lower_theta = self._height2theta(height)
        
        _chord = 2 * np.sqrt(self.D * height - height**2)

        return lower_theta * self.R**2 / 2 - (self.R - height) * _chord / 2

    def major(self, upper:bool = True) -> bool:
        """
        bool : True if the segment is a major segment (h >= R), False otherwise.

        Notes
        -----
        - Upper segment is major when h < R
        - Lower segment is major when h >= R
        - At h = R, both segments are semicircles; this implementation classifies
          the lower segment as major and the upper segment as not major
        """
        flag = self.h >= self.R

        return not flag if upper else flag