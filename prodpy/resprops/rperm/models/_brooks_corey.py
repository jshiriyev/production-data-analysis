from dataclasses import dataclass
from typing import Union

import numpy as np

ArrayLike = Union[float, list[float], np.ndarray]

@dataclass
class BrooksCorey:
    """
    Brooks-Corey saturation relation combined with the Burdine relative
    permeability model.

    The model uses wetting-phase saturation Sw and computes:

        Sw* = (Sw - Swr) / (1 - Swr - Snwr)

        krw  = krw0  * Sw*^(2/lambda + 3)

        krnw = krnw0 * (1 - Sw*)^2 * (1 - Sw*^(2/lambda + 1))

    Parameters
    ----------
    Swr : float
        Residual/irreducible wetting-phase saturation.
    Snwr : float
        Residual non-wetting-phase saturation.
    krw0 : float
        Endpoint wetting-phase relative permeability (@Snwr).
    krnw0 : float
        Endpoint non-wetting-phase relative permeability (@Swr).
    lambda_ : float
        Brooks-Corey pore-size distribution index.
    """
    Swr: float
    Snwr: float
    krw0: float
    krnw0: float
    lambda_: float

    def __post_init__(self) -> None:
        """Validate model parameters after initialization."""

        if not 0 <= self.Swr < 1:
            raise ValueError("Swr must be in the range [0, 1).")

        if not 0 <= self.Snwr < 1:
            raise ValueError("Snwr must be in the range [0, 1).")

        if self.Swr + self.Snwr >= 1:
            raise ValueError("Swr + Snwr must be less than 1.")

        if not 0 <= self.krw0 <= 1:
            raise ValueError("krw0 must be in the range [0, 1].")

        if not 0 <= self.krnw0 <= 1:
            raise ValueError("krnw0 must be in the range [0, 1].")

        if self.lambda_ <= 0:
            raise ValueError("lambda_ must be positive.")
        
    @property
    def mobile_saturation_range(self) -> float:
        """Return the mobile saturation interval: 1 - Swr - Snwr."""
        return 1.0 - self.Swr - self.Snwr
        
    @property
    def Sw_min(self) -> float:
        """Minimum physically valid wetting-phase saturation."""
        return self.Swr

    @property
    def Sw_max(self) -> float:
        """Maximum physically valid wetting-phase saturation."""
        return 1.0 - self.Snwr
    
    @property
    def nw(self) -> float:
        """Wetting-phase exponent on relative permeability curve."""
        return 2.0 / self.lambda_ + 3.0
    
    @property
    def nnw(self) -> float:
        """Non-wetting-phase exponent on relative permeability curve."""
        return 2.0 / self.lambda_ + 1.0

    def Swd(self, Sw:ArrayLike, clip:bool=True):
        """
        Return dimensionless / normalized wetting-phase saturation.

        Parameters
        ----------
        Sw : float, list, or numpy.ndarray
            Wetting-phase saturation.
        clip : bool, default=True
            If True, Sw values are clipped to the physical range
            [Swr, 1 - Snwr]. If False, values outside this range raise
            a ValueError.

        Returns
        -------
        numpy.ndarray
            Normalized wetting-phase saturation.
        """
        Sw_array = np.asarray(Sw, dtype=float)

        if clip:
            Sw_array = np.clip(Sw_array, self.Swr, 1-self.Snwr)
        else:
            if np.any((Sw_array < self.Sw_min) | (Sw_array > self.Sw_max)):
                raise ValueError(
                    f"Sw must be between {self.Sw_min:.4f} and {self.Sw_max:.4f}. "
                    "Use clip=True to automatically clip values."
                )

        return (Sw_array-self.Swr)/self.mobile_saturation_range

    def get(self, Sw:ArrayLike, clip:bool=True):
        """Computes relative permeabilities at a given wetting phase saturation.

        Parameters:
        ----------
        Sw   : numpy array
            wetting phase saturation.
        clip : bool, default=True
            Whether to clip Sw to the physically valid saturation range.

        Returns:
        -------
        krw  : numpy array
            wetting phase relative permeability.
        krnw : numpy array
            non-wetting phase relative permeability.

        """
        S = self.Swd(Sw, clip=clip)

        krw = self.krw0 * S**self.nw
        krnw = self.krnw0 * (1.0 - S) ** 2 * (1.0 - S**self.nnw)

        return krw, krnw

    @staticmethod
    def mobility_ratio(
        krw:ArrayLike,
        krnw:ArrayLike,
        muw:ArrayLike,
        munw:ArrayLike
    ) -> np.ndarray:
        """
        Compute the mobility ratio.

        M = (krw / muw) / (krnw / munw)

        Parameters
        ----------
        krw : float, list, or numpy.ndarray
            Wetting-phase relative permeability.
        krnw : float, list, or numpy.ndarray
            Non-wetting-phase relative permeability.
        muw : float, list, or numpy.ndarray
            Wetting-phase viscosity.
        munw : float, list, or numpy.ndarray
            Non-wetting-phase viscosity.

        Returns
        -------
        numpy.ndarray
            Mobility ratio.
        """
        krw_array = np.asarray(krw, dtype=float)
        krnw_array = np.asarray(krnw, dtype=float)
        muw_array = np.asarray(muw, dtype=float)
        munw_array = np.asarray(munw, dtype=float)

        if np.any(muw_array <= 0):
            raise ValueError("muw must be positive.")

        if np.any(munw_array <= 0):
            raise ValueError("munw must be positive.")

        mobility_w = krw_array/muw_array  # Wetting phase mobility
        mobility_nw = krnw_array/munw_array  # Non-wetting phase mobility

        return np.divide(
            mobility_w,
            mobility_nw,
            out=np.full_like(mobility_w, np.inf, dtype=float),
            where=mobility_nw != 0,
        )