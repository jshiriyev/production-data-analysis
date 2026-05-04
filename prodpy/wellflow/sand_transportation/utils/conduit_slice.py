from __future__ import annotations

from dataclasses import dataclass

from .circular_conduit import CircularConduit
from .annular_conduit import AnnularConduit

@dataclass(slots=True, frozen=True)
class ConduitSlice:
    """
    Represent one axial slice of a conduit.

    Parameters
    ----------
    upper  : float
        Upper depth of the slice, upstream.
    lower  : float
        Lower depth of the slice, downstream.

    Notes
    -----
    The slice length is computed as ``lower - upper`` and must be non-negative.
    The slice center is computed as ``0.5 * (upper + lower)``.
        
    """
    upper   : float
    lower   : float

    profile : CircularConduit | AnnularConduit | None = None

    def __post_init__(self) -> None:
        """Validate slice bounds after initialization."""
        if self.lower <= self.upper:
            raise ValueError(
                f"'lower' must be greater than 'upper'. "
                f"Got upper={self.upper}, lower={self.lower}."
            )
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"  - Upper depth of the slice, upstream: {self.upper:.3f}\n"
            f"  - Lower depth of the slice, downstream: {self.lower:.3f}\n\n"
            f"Cross sectional profile: {self.profile.__repr__()}"
        )
    
    def __call__(self, add_volume:float, **kwargs) -> float:

        self.profile(add_volume/self.length, **kwargs)

    @property
    def length(self):
        """Return the slice length."""
        return self.lower - self.upper
    
    @property
    def center(self):
        """Return the center depth of the slice."""
        return 0.5 * (self.upper + self.lower)