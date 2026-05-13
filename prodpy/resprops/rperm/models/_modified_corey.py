from dataclasses import dataclass
from typing import Union

import numpy as np

ArrayLike = Union[float, list[float], np.ndarray]

@dataclass
class ModifiedCorey:
    """
    Modified Corey relative permeability model for two-phase flow.

    The model uses normalized wetting-phase saturation:

        Sw* = (Sw - Swr) / (1 - Swr - Snwr)

    and computes:

        krw  = krw0  * Sw*^nw

        krnw = krnw0 * (1 - Sw*)^nnw

    Parameters
    ----------
    Swr   : float
        Residual wetting-phase saturation.
    Snwr  : float
        Residual non-wetting-phase saturation.
    krw0  : float
        Endpoint wetting-phase relative permeability.
    krnw0 : float
        Endpoint non-wetting-phase relative permeability.
    nw    : float
        Wetting-phase Corey exponent.
    nnw   : float
        Non-wetting-phase Corey exponent.

    """
    Swr: float = 0.2
    Snwr: float = 0.2
    krw0: float = 0.2
    krnw0: float = 0.2
    nw: float = 2.0
    nnw: float = 2.0

    def __post_init__(self) -> None:
        """Validate model parameters."""

        if not 0 <= self.Swr < 1:
            raise ValueError("Swr must be in the range [0, 1).")

        if not 0 <= self.Snwr < 1:
            raise ValueError("Snwr must be in the range [0, 1).")

        if self.Swr + self.Snwr >= 1:
            raise ValueError("Swr + Snwr must be less than 1.")

        if self.krw0 < 0:
            raise ValueError("krw0 must be non-negative.")

        if self.krnw0 < 0:
            raise ValueError("krnw0 must be non-negative.")

        if self.nw <= 0:
            raise ValueError("nw must be positive.")

        if self.nnw <= 0:
            raise ValueError("nnw must be positive.")
        
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"Swr={self.Swr}, "
            f"Snwr={self.Snwr}, "
            f"krw0={self.krw0}, "
            f"krnw0={self.krnw0}, "
            f"nw={self.nw}, "
            f"nnw={self.nnw}"
            f")"
        )
        
    @property
    def mobile_saturation_range(self) -> float:
        """Return mobile saturation interval: 1 - Swr - Snwr."""
        return 1.0 - self.Swr - self.Snwr

    @property
    def Sw_min(self) -> float:
        """Minimum valid wetting-phase saturation."""
        return self.Swr

    @property
    def Sw_max(self) -> float:
        """Maximum valid wetting-phase saturation."""
        return 1.0 - self.Snwr

    def Swd(self, Sw: ArrayLike, clip: bool = True) -> np.ndarray:
        """
        Return normalized wetting-phase saturation.

        Parameters
        ----------
        Sw : float, list, or numpy.ndarray
            Wetting-phase saturation.
        clip : bool, default=True
            If True, values are clipped to [Swr, 1 - Snwr].
            If False, values outside this range raise a ValueError.

        Returns
        -------
        numpy.ndarray
            Normalized wetting-phase saturation.
        """
        Sw_array = np.asarray(Sw, dtype=float)

        if clip:
            Sw_array = np.clip(Sw_array, self.Sw_min, self.Sw_max)
        else:
            if np.any((Sw_array < self.Sw_min) | (Sw_array > self.Sw_max)):
                raise ValueError(
                    f"Sw must be between {self.Sw_min:.4f} and {self.Sw_max:.4f}. "
                    "Use clip=True to automatically clip values."
                )

        return (Sw_array - self.Swr) / self.mobile_saturation_range

    def get(self, Sw: ArrayLike, clip: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute wetting and non-wetting relative permeabilities.

        Parameters
        ----------
        Sw : float, list, or numpy.ndarray
            Wetting-phase saturation.
        clip : bool, default=True
            Whether to clip Sw to the valid saturation range.

        Returns
        -------
        krw : numpy.ndarray
            Wetting-phase relative permeability.
        krnw : numpy.ndarray
            Non-wetting-phase relative permeability.

        """
        S = self.Swd(Sw, clip=clip)
        
        krw = self.krw0 * S**self.nw
        krnw = self.krnw0 * (1 - S)**self.nnw

        return krw,krnw

if __name__ == "__main__":

    rp = ModifiedCorey(0.1,0.4,0.3,0.8)

    print(rp.get(0.3))

    rp = ModifiedCorey(0.3,0.05,0.8,0.3)

    print(rp.get(0.8))