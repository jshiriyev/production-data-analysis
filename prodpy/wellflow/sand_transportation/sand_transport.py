import numpy as np

import numbers

from .utils import ConduitSlice, AnnularConduit

class SandTransport:
    """
    Computational grid and state container for sand transport and settling
    in a concentric annulus.

    Parameters
    ----------
    well_length : float
        Total well length [m].
    outer_diameter : float
        Outer diameter of the annulus [m].
    inner_diameter : float
        Inner diameter of the annulus [m].
    slice_count : int
        Number of axial computational slices.

    Notes
    -----
    - The well is discretized uniformly from 0 to `well_length`.
    - Each slice can be accessed using indexing, e.g. `model[0]`.
    - `sand_height[i]` represents the deposited sand height in slice `i`.
    """

    _MODEL_NAME = "Sand Transport and Settling Model"

    _ASSUMPTIONS = {
        "flow_regime": "single-phase liquid",
        "geometry": "concentric annulus",
        "flow_condition": "steady-state",
        "fluid_model": "Newtonian fluid",
        "fluid_properties": "constant density and viscosity",
        "particle_size_model": "single representative particle size",
        "particle_shape": "spherical particles",
        "particle_density": "constant",
        "particle_concentration": "dilute suspension",
        "velocity_model": "cross-sectional average annular velocity",
        "settling_model": "terminal settling velocity with drag law",
        "bed_model": "not included",
        "wall_effects": "neglected",
        "turbulent_dispersion": "neglected",
        "sand_generation": "external to transport model",
    }

    def __init__(self, well_length:float, outer_diameter:float, inner_diameter: float, slice_count:int) -> None:

        self._well_length = float(well_length)
        self._outer_diameter = float(outer_diameter)
        self._inner_diameter = float(inner_diameter)
        self._slice_count = slice_count

        self._validate_inputs()

        self._slice_edges = self._build_slice_edges()
        self._sand_height = np.zeros(self.slice_count, dtype=float)

    def _validate_inputs(self) -> None:
        if self._outer_diameter <= 0:
            raise ValueError("outer_diameter must be > 0.")
        if self._inner_diameter <= 0:
            raise ValueError("inner_diameter must be > 0.")
        if self._outer_diameter <= self._inner_diameter:
            raise ValueError("outer_diameter must be greater than inner_diameter.")
        if self._well_length <= 0:
            raise ValueError("well_length must be > 0.")
        if not isinstance(self._slice_count, numbers.Integral):
            raise TypeError("slice_count must be an integer.")
        if self._slice_count <= 0:
            raise ValueError("slice_count must be > 0.")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"  - The lenght of the well: {self.well_length}\n"
            f"  - The inner diameter of the outer pipe: {self.outer_diameter}\n"
            f"  - The outer diameter of the inner pipe: {self.inner_diameter}\n"
            f"  - The slice count: {self.slice_count}"
        )
    
    def describe(self) -> str:
        assumptions_text = "\n".join(
            f"  - {key}: {value}" for key, value in self._ASSUMPTIONS.items()
        )
        return (
            f"{self._MODEL_NAME}\n"
            f"  - the lenght of the well : {self.well_length}\n"
            f"  - the inner diameter of the outer pipe : {self.outer_diameter}\n"
            f"  - the outer diameter of the inner pipe : {self.inner_diameter}\n"
            f"  - the slice count : {self.slice_count}\n"
            f"  - the slice length (uniform size distribution) : {self.slice_length:.3f}\n\n"
            f"Assumptions in the model :\n{assumptions_text}"
        )
    
    def __len__(self) -> int:
        return self.slice_count
    
    def __getitem__(self, index:int) -> ConduitSlice:
        if not isinstance(index, numbers.Integral):
            raise TypeError("Slice index must be an integer.")
    
        index = index if index >= 0 else index + self.slice_count

        if not 0 <= index < self.slice_count:
            raise IndexError(f"Slice index out of range: {index}")
        
        return ConduitSlice(
            upper = self._slice_edges[index],
            lower = self._slice_edges[index+1],
            profile = AnnularConduit(
                self._outer_diameter,
                self._inner_diameter,
                self._sand_height[index])
        )

    def __iter__(self):
        for i in range(self.slice_count):
            yield self[i]

    def _build_slice_edges(self) -> np.ndarray:
        """Return uniformly spaced slice edges from 0 to well_length."""
        return np.linspace(0.0, self.well_length, self.slice_count + 1)

    @property
    def model_name(self) -> str:
        return self._MODEL_NAME.copy()

    @property
    def assumptions(self) -> dict:
        return self._ASSUMPTIONS.copy()

    @property
    def well_length(self) -> float:
        return self._well_length
    
    @property
    def outer_diameter(self) -> float:
        return self._outer_diameter
    
    @property
    def inner_diameter(self) -> float:
        return self._inner_diameter
    
    @property
    def slice_count(self) -> int:
        return self._slice_count
    
    @property
    def slice_length(self) -> float:
        return self.well_length / self.slice_count

    @property
    def slice_edges(self) -> np.ndarray:
        return self._slice_edges.copy()

    @property
    def sand_height(self) -> np.ndarray:
        return self._sand_height.copy()

def fluid_velocity(Q: float, A: float) -> float:
    """
    Calculate fluid velocity from volumetric flow rate and cross-sectional area.

    This is a generic form of the continuity equation:

        u = Q / A

    where:
        u = velocity [m/s]
        Q = volumetric flow rate [m^3/s]
        A = flow area [m^2]

    Parameters
    ----------
    Q : float
        Volumetric flow rate [m^3/s].
        Must be non-negative.

    A : float
        Cross-sectional flow area [m^2].
        Must be strictly positive.

    Returns
    -------
    float
        Fluid velocity [m/s].

    Raises
    ------
    ValueError
        If A <= 0 or Q < 0.

    Notes
    -----
    - This function is geometry-independent (pipe, annulus, irregular shape).
    - For annular flow, area should be precomputed as:
          A = (π/4) * (D_o^2 - D_i^2)
    - Useful for coupling with particle transport models:
          u_p = u - vs

    Examples
    --------
    >>> velocity_from_area(0.02, 0.01)
    2.0

    >>> velocity_from_area(Q=0.05, A=0.02)
    2.5
    """

    if A <= 0:
        raise ValueError("Area must be positive.")
    if Q < 0:
        raise ValueError("Flow rate must be non-negative.")

    return Q / A

def particle_velocity(vf, vs):
    return vf - vs

def transport_flag(vf, vc):
    return True if vf >= vc else False
