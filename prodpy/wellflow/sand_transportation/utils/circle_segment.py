from dataclasses import dataclass

import math

@dataclass
class CircleSegment:
    """
    Generic container for properties of a segment.

    Note
    ----
    This class does NOT enforce any geometric relationships between attributes.
    The values are treated as independent inputs.

    Parameters
    ----------
    chord : float
        Boundary length of the segment [units of length]
    arc : float
        Arc length of the segment [units of length]
    area : float
        Area of the segment [units of length^2]
    major : bool
        True if the segment is considered a major segment, False otherwise
    """

    chord   : float
    arc     : float
    area    : float
    major   : bool

    def __post_init__(self):
        self._validate()

    def _validate(self):

        self._validate_positive_finite(self.chord, "Chord length")
        self._validate_positive_finite(self.arc, "Arc length")
        self._validate_positive_finite(self.area, "Area")
        
        if not isinstance(self.major, bool):
            raise TypeError("major must be a boolean")
        
    @property
    def perimeter(self) -> float:
        """
        Full perimeter of the segment.

        Returns
        -------
        float
            Total boundary length of the segment, equal to arc length plus chord length.
        """
        return self.chord + self.arc
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"  - The length of inner boundary separating the segment: {self.chord:.3f}\n"
            f"  - The length of outer boundary separating the segment, arc length: {self.arc:.3f}\n"
            f"  - The area of the segment: {self.area:.3f}\n"
            f"  - The perimeter of the segment, sum of the inner and outer boundaries: {self.perimeter:.3f}\n"
            f"  - The hydarulic mean diameter: {self.Dh:.3f}\n"
            f"  - The type for the segment: {'major' if self.major else 'minor'}"
        )
    
    @property
    def Dh(self) -> float:
        return 0. if self.perimeter == 0 else 4 * self.area / self.perimeter
    
    @property
    def hydraulic_diameter(self) -> float:
        return self.Dh
    
    @property
    def minor(self) -> bool:
        return not self.major
    
    @staticmethod
    def _validate_positive_finite(value: float, name: str) -> None:

        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric.")

        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite (not NaN or inf).")

        if value < 0:
            raise ValueError(f"{name} must be positive.")