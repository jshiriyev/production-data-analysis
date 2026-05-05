from dataclasses import dataclass
from typing import Optional, Union, Dict
import math

import numpy as np

Number = Union[int, float, np.ndarray]

@dataclass(frozen=True)
class Boundary:
    """Dietz drainage-area shape-factor record."""
    #shape factor, {C_A} value
    shape_factor: float
    # Use infinite system solution
    tDA_infinite_acting_end: Optional[float]
    # PSS is exact for higher values
    tDA_pss_start: Optional[float]

    note: str = ""

class PorousMedia():
    """A reservoir class for flow calculations."""
    EULER_GAMMA = 0.5772156649015329

    FT_TO_METER = 0.3048
    ACRE_TO_SQFT = 43560

    SHAPE_FACTORS: Dict[str, Boundary] = {
        "circle_center":       Boundary(31.62,   0.10,  0.06, "Circle, centered well"),
        "hexagon_center":      Boundary(31.6,    0.10,  0.06, "Hexagon, centered well"),
        "triangle_center":     Boundary(27.6,    0.09,  0.07, "Triangle, centered well"),
        "parallelogram_center":Boundary(27.1,    0.09,  0.07, "Parallelogram/rhombus row"),
        "triangle_row":        Boundary(21.9,    0.08,  0.12, "Triangle row; verify well location"),
        "triangle_low_CA":     Boundary( 0.098,  0.015, 0.60, "Low-C_A triangular/boundary row"),
        "square_center":       Boundary(30.8828, 0.09,  0.05, "Square, centered well"),
        "square_quadrants":    Boundary(12.9851, 0.03,  0.25, "Square subdivided row"),
        "square_offcenter":    Boundary( 4.5132, 0.025, 0.30, "Square off-center row"),
        "square_grid":         Boundary( 3.3351, 0.01,  0.25, "Square grid row"),
        "rect_center":         Boundary(21.8369, 0.025, 0.15, "Rectangle centered row"),
        "rect_split_center":   Boundary(10.8374, 0.025, 0.15, "Rectangle split/center row"),
        "rect_offcenter":      Boundary( 4.5141, 0.06,  0.50, "Rectangle off-center row"),
        "rect_corner":         Boundary( 2.0769, 0.02,  0.50, "Rectangle/corner row"),
        "long_rect_center":    Boundary( 3.1573, 0.005, 0.15, "Long rectangle row"),
        "long_rect_edge":      Boundary( 0.5813, 0.02,  0.60, "Long rectangle edge row"),
        "very_long_rect_edge": Boundary( 0.1109, 0.005, 0.60, "Very long rectangle edge row"),
        "strip_center":        Boundary( 5.3790, 0.01,  0.30, "Strip/rectangle centered row"),
        "strip_split":         Boundary( 2.6896, 0.01,  0.30, "Strip split row"),
        "strip_offcenter":     Boundary( 0.2318, 0.03,  2.00, "Strip off-center row"),
        "strip_edge":          Boundary( 0.1155, 0.01,  2.00, "Strip edge row"),
    }

    def __init__(
        self,
        area: float,
        height: float = 1.,
        /,
        shape_key: str = "circle_center",
        shape_factor: Optional[float] = None,
        tDA_infinite_acting_end: Optional[float] = None,
        tDA_pss_start: Optional[float] = None,
        note: str = "Circle, centered well",
    ) -> None:
        """Initialize the base reservoir class for porous mediaflow calculations.

        Arguments:
        ---------
        area (float): in acre.
        height (float): in ft, defaults to 1 ft.

        """
        self.area = area
        self.height = height

        if shape_factor is not None:
            self.shape_key = "custom"
            self.shape = Boundary(
                shape_factor = self._validate_positive("shape factor", shape_factor),
                tDA_infinite_acting_end = tDA_infinite_acting_end,
                tDA_pss_start = tDA_pss_start,
                note = note,
            )
        else:
            if shape_key not in self.SHAPE_FACTORS:
                raise KeyError(f"Unknown shape_key: {shape_key}")
            self.shape_key = shape_key
            self.shape = self.SHAPE_FACTORS[shape_key]

    @staticmethod
    def _validate_positive(name: str, value: float) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a positive finite value.") from None

        if value <= 0 or not math.isfinite(value):
            raise ValueError(f"{name} must be a positive finite value.")
        return value

    @classmethod
    def available_shape_factors(cls) -> Dict[str, Boundary]:
        """Return available shape-factor records."""
        return cls.SHAPE_FACTORS.copy()

    @property
    def area(self):
        """Getter for the reservoir area."""
        return self._area / self.FT_TO_METER**2 / self.ACRE_TO_SQFT

    @area.setter
    def area(self,value):
        """Setter for the reservoir area."""
        value = self._validate_positive("area", value)
        self._area = value * self.ACRE_TO_SQFT * self.FT_TO_METER**2

    @property
    def height(self):
        """Getter for the reservoir height."""
        return self._height / self.FT_TO_METER

    @height.setter
    def height(self,value):
        """Setter for the reservoir height."""
        value = self._validate_positive("height", value)
        self._height = value * self.FT_TO_METER

    @property
    def radius_equivalent(self) -> float:
        """Circular equivalent drainage radius."""
        return self._radius_equivalent / self.FT_TO_METER
    
    @property
    def _radius_equivalent(self) -> float:
        """Setter for the reservoir equivalent radius."""
        return math.sqrt(self._area / math.pi)

    @property
    def shape_factor(self) -> float:
        return self.shape.shape_factor

    @property
    def volume(self) -> float:
        """Getter for the reservoir volume."""
        return self._volume/(self.FT_TO_METER**3)

    @property
    def _volume(self) -> float:
        """Setter for the reservoir volume."""
        return self._area*self._height

if __name__ == "__main__":

    res = PorousMedia(1000,20)

    print(res.area)
    print(res._area)
    print(res.radius_equivalent)
    print(res.volume)