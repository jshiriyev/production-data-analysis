from dataclasses import dataclass, field
from typing import NamedTuple, Optional, Union

import numpy as np

from ._modified_corey import ModifiedCorey

ArrayLike = Union[float, list[float], np.ndarray]

class ThreePhaseRelPerm(NamedTuple):
    """
    Container for three-phase relative permeability results.
    """
    krw: np.ndarray
    kro: np.ndarray
    krg: np.ndarray

@dataclass
class StoneModelI:
    """
    Stone I three-phase relative permeability model.

    This class estimates three-phase water, oil, and gas relative permeabilities
    using two two-phase relative permeability models:

    1. Oil-water system:
           krw(sw), kro_ow(sw)

    2. Gas-oil system:
           kro_go(sl), krg(sl)

       where:

           sl = sw + so

       is the liquid saturation.

    The three-phase oil relative permeability is calculated using Stone's
    Model I:

        kro = so_star * kro_ow * kro_go / (
                  (1 - sw_star) * (1 - sg_star) * kro0_ow
              )

    where sw_star, so_star, and sg_star are normalized three-phase
    saturations.

    Parameters
    ----------
    swr : float
        Residual / irreducible water saturation.
    sor_ow : float
        Residual oil saturation in the oil-water system.
    sor_go : float
        Residual oil saturation in the gas-oil system.
    sgr : float
        Critical / residual gas saturation.
    krw0 : float
        Endpoint water relative permeability in the oil-water system.
    kro0_ow : float
        Endpoint oil relative permeability in the oil-water system.
    kro0_go : float
        Endpoint oil relative permeability in the gas-oil system.
    krg0 : float
        Endpoint gas relative permeability in the gas-oil system.
    nw : float
        Water Corey exponent in the oil-water system.
    no_ow : float
        Oil Corey exponent in the oil-water system.
    no_go : float
        Oil Corey exponent in the gas-oil system.
    ng : float
        Gas Corey exponent in the gas-oil system.
    method : str
        Method for estimating minimum oil saturation in the three-phase system.

        Supported values are:

        - "minimum":
              som = min(sor_ow, sor_go)

        - "average":
              som is interpolated between sor_ow and sor_go as a function of
              gas saturation.
    """

    swr: float = 0.1
    sor_ow: float = 0.4
    sor_go: float = 0.4
    sgr: float = 0.05

    krw0: float = 0.3
    kro0_ow: float = 0.8
    kro0_go: float = 0.8
    krg0: float = 0.3

    nw: float = 2.0
    no_ow: float = 2.0
    no_go: float = 2.0
    ng: float = 2.0

    method: str = "average"

    oil_water : ModifiedCorey = field(init=False)
    gas_oil   : ModifiedCorey = field(init=False)

    def __post_init__(self) -> None:
        """
        Validate inputs and initialize the two-phase relative permeability models.
        """

        self._validate_parameters()

        self.oil_water = ModifiedCorey(
            Swr=self.swr,
            Snwr=self.sor_ow,
            krw0=self.krw0,
            krnw0=self.kro0_ow,
            nw=self.nw,
            no=self.no_ow,
        )

        self.gas_oil = ModifiedCorey(
            Swr=self.slr_go,
            Snwr=self.sgr,
            krw0=self.kro0_go,
            krnw0=self.krg0,
            nw=self.no_go,
            no=self.ng,
        )

    def _validate_parameters(self) -> None:
        """
        Validate model parameters.
        """

        saturation_params = {
            "swr": self.swr,
            "sor_ow": self.sor_ow,
            "sor_go": self.sor_go,
            "sgr": self.sgr,
        }

        for name, value in saturation_params.items():
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in the range [0, 1).")

        if self.swr + self.sor_ow >= 1.0:
            raise ValueError("swr + sor_ow must be less than 1.")

        if self.swr + self.sor_go + self.sgr >= 1.0:
            raise ValueError("swr + sor_go + sgr must be less than 1.")

        if self.swr + self.sor_ow + self.sgr >= 1.0:
            raise ValueError("swr + sor_ow + sgr must be less than 1.")

        kr_endpoints = {
            "krw0": self.krw0,
            "kro0_ow": self.kro0_ow,
            "kro0_go": self.kro0_go,
            "krg0": self.krg0,
        }

        for name, value in kr_endpoints.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in the range [0, 1].")

        exponents = {
            "nw": self.nw,
            "no_ow": self.no_ow,
            "no_go": self.no_go,
            "ng": self.ng,
        }

        for name, value in exponents.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive.")

        valid_methods = {"minimum", "average"}

        if self.method not in valid_methods:
            raise ValueError(
                f"Invalid method '{self.method}'. "
                f"Choose from {sorted(valid_methods)}."
            )

    @property
    def slr_go(self) -> float:
        """
        Residual liquid saturation in the gas-oil system.

        Formula
        -------
        slr_go = swr + sor_go
        """
        return self.swr + self.sor_go

    @property
    def ow(self) -> ModifiedCorey:
        """
        Short alias for the oil-water two-phase relative permeability model.
        """
        return self.oil_water

    @property
    def go(self) -> ModifiedCorey:
        """
        Short alias for the gas-oil two-phase relative permeability model.
        """
        return self.gas_oil

    @staticmethod
    def _as_array(value: ArrayLike, name: str) -> np.ndarray:
        """
        Convert an input value to a NumPy array.
        """
        try:
            return np.asarray(value, dtype=float)
        except Exception as exc:
            raise TypeError(f"{name} must be a scalar, list, or numpy array.") from exc

    @staticmethod
    def _clip01(values: np.ndarray, clip: bool = True) -> np.ndarray:
        """
        Clip values to [0, 1] if requested.
        """
        if clip:
            return np.clip(values, 0.0, 1.0)

        return values

    def swd(self, sw: ArrayLike, clip: bool = True) -> np.ndarray:
        """
        Return normalized water saturation for the oil-water system.

        Formula
        -------
        swd = (sw - swr) / (1 - swr - sor_ow)

        Parameters
        ----------
        sw : float or array-like
            Water saturation.
        clip : bool, default=True
            If True, returned values are clipped to [0, 1].

        Returns
        -------
        numpy.ndarray
            Normalized water saturation.
        """
        return self.ow.Swd(sw, clip=clip)

    def sod(self, so: ArrayLike, clip: bool = True) -> np.ndarray:
        """
        Return normalized oil saturation.

        Formula
        -------
        sod = (so - sor_ow) / (1 - swr - sor_ow - sgr)

        Parameters
        ----------
        so : float or array-like
            Oil saturation.
        clip : bool, default=True
            If True, returned values are clipped to [0, 1].

        Returns
        -------
        numpy.ndarray
            Normalized oil saturation.
        """
        so_array = self._as_array(so, "so")
        values = (so_array - self.sor_ow) / (
            1.0 - self.swr - self.sor_ow - self.sgr
        )

        return self._clip01(values, clip=clip)

    def sgd(self, sg: ArrayLike, clip: bool = True) -> np.ndarray:
        """
        Return normalized gas saturation for the gas-oil system.

        Formula
        -------
        sgd = (sg - sgr) / (1 - sgr - slr_go)

        Since:

            slr_go = swr + sor_go

        the denominator becomes:

            1 - sgr - swr - sor_go

        Parameters
        ----------
        sg : float or array-like
            Gas saturation.
        clip : bool, default=True
            If True, returned values are clipped to [0, 1].

        Returns
        -------
        numpy.ndarray
            Normalized gas saturation.
        """
        sg_array = self._as_array(sg, "sg")
        values = (sg_array - self.sgr) / (1.0 - self.sgr - self.slr_go)

        return self._clip01(values, clip=clip)

    def som(self, sg: Optional[ArrayLike] = None) -> Union[float, np.ndarray]:
        """
        Return minimum oil saturation in the three-phase system.

        The value is estimated using one of two methods.

        method = "minimum"
            som = min(sor_ow, sor_go)

        method = "average"
            som is interpolated between sor_ow and sor_go using normalized
            mobile gas saturation:

                a = (sg - sgr) / (1 - swr - sor_go - sgr)

                som = (1 - a) * sor_ow + a * sor_go

        Parameters
        ----------
        sg : float or array-like, optional
            Gas saturation. Required when method="average".

        Returns
        -------
        float or numpy.ndarray
            Minimum oil saturation in the three-phase system.
        """

        if self.method == "minimum":
            return min(self.sor_ow, self.sor_go)

        if self.method == "average":
            if sg is None:
                raise ValueError("Gas saturation sg must be provided for method='average'.")

            sg_array = self._as_array(sg, "sg")

            denominator = 1.0 - self.swr - self.sor_go - self.sgr

            a = (sg_array - self.sgr) / denominator
            a = np.clip(a, 0.0, 1.0)

            return (1.0 - a) * self.sor_ow + a * self.sor_go

        raise ValueError(
            f"Invalid method '{self.method}'. Choose from ['minimum', 'average']."
        )

    def sw_star(
        self,
        sw: ArrayLike,
        som: Union[float, np.ndarray],
        clip: bool = True,
    ) -> np.ndarray:
        """
        Return normalized water saturation for Stone I.

        Formula
        -------
        sw_star = (sw - swr) / (1 - swr - som)

        Parameters
        ----------
        sw : float or array-like
            Water saturation.
        som : float or array-like
            Minimum oil saturation in the three-phase system.
        clip : bool, default=True
            If True, returned values are clipped to [0, 1].

        Returns
        -------
        numpy.ndarray
            Normalized water saturation.
        """
        sw_array = self._as_array(sw, "sw")
        som_array = np.asarray(som, dtype=float)

        values = (sw_array - self.swr) / (1.0 - self.swr - som_array)

        return self._clip01(values, clip=clip)

    def so_star(
        self,
        so: ArrayLike,
        som: Union[float, np.ndarray],
        clip: bool = True,
    ) -> np.ndarray:
        """
        Return normalized oil saturation for Stone I.

        Formula
        -------
        so_star = (so - som) / (1 - swr - som)

        Parameters
        ----------
        so : float or array-like
            Oil saturation.
        som : float or array-like
            Minimum oil saturation in the three-phase system.
        clip : bool, default=True
            If True, returned values are clipped to [0, 1].

        Returns
        -------
        numpy.ndarray
            Normalized oil saturation.
        """
        so_array = self._as_array(so, "so")
        som_array = np.asarray(som, dtype=float)

        values = (so_array - som_array) / (1.0 - self.swr - som_array)

        return self._clip01(values, clip=clip)

    def sg_star(
        self,
        sg: ArrayLike,
        som: Union[float, np.ndarray],
        clip: bool = True,
    ) -> np.ndarray:
        """
        Return normalized gas saturation for Stone I.

        Formula
        -------
        sg_star = sg / (1 - swr - som)

        Parameters
        ----------
        sg : float or array-like
            Gas saturation.
        som : float or array-like
            Minimum oil saturation in the three-phase system.
        clip : bool, default=True
            If True, returned values are clipped to [0, 1].

        Returns
        -------
        numpy.ndarray
            Normalized gas saturation.
        """
        sg_array = self._as_array(sg, "sg")
        som_array = np.asarray(som, dtype=float)

        values = sg_array / (1.0 - self.swr - som_array)

        return self._clip01(values, clip=clip)

    def kro(
        self,
        sw: ArrayLike,
        so: ArrayLike,
        sg: ArrayLike,
        kro_ow: ArrayLike,
        kro_go: ArrayLike,
        clip: bool = True,
    ) -> np.ndarray:
        """
        Return three-phase oil relative permeability using Stone I.

        Formula
        -------
        kro = so_star * kro_ow * kro_go / (
                  (1 - sw_star) * (1 - sg_star) * kro0_ow
              )

        Parameters
        ----------
        sw : float or array-like
            Water saturation.
        so : float or array-like
            Oil saturation.
        sg : float or array-like
            Gas saturation.
        kro_ow : float or array-like
            Oil relative permeability from the oil-water two-phase curve.
        kro_go : float or array-like
            Oil relative permeability from the gas-oil two-phase curve.
        clip : bool, default=True
            If True, final oil relative permeability is clipped to
            [0, kro0_ow].

        Returns
        -------
        numpy.ndarray
            Three-phase oil relative permeability.
        """

        sw_array = self._as_array(sw, "sw")
        so_array = self._as_array(so, "so")
        sg_array = self._as_array(sg, "sg")

        kro_ow_array = self._as_array(kro_ow, "kro_ow")
        kro_go_array = self._as_array(kro_go, "kro_go")

        som = self.som(sg_array)

        sw_star = self.sw_star(sw_array, som)
        so_star = self.so_star(so_array, som)
        sg_star = self.sg_star(sg_array, som)

        upper = so_star * kro_ow_array * kro_go_array

        lower = (
            (1.0 - sw_star)
            * (1.0 - sg_star)
            * self.kro0_ow
        )

        kro = np.divide(
            upper,
            lower,
            out=np.zeros_like(upper, dtype=float),
            where=lower > 0.0,
        )

        if clip:
            kro = np.clip(kro, 0.0, self.kro0_ow)

        return kro

    def _prepare_saturations(
        self,
        sw: Optional[ArrayLike],
        so: Optional[ArrayLike],
        sg: Optional[ArrayLike],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert input saturations to arrays and calculate the missing saturation.

        At least two of sw, so, and sg must be provided. The missing saturation
        is calculated from:

            sw + so + sg = 1

        Parameters
        ----------
        sw : float or array-like, optional
            Water saturation.
        so : float or array-like, optional
            Oil saturation.
        sg : float or array-like, optional
            Gas saturation.

        Returns
        -------
        sw : numpy.ndarray
            Water saturation.
        so : numpy.ndarray
            Oil saturation.
        sg : numpy.ndarray
            Gas saturation.
        """

        provided_count = sum(value is not None for value in (sw, so, sg))

        if provided_count < 2:
            raise ValueError("At least two phase saturations must be provided.")

        sw_array = None if sw is None else self._as_array(sw, "sw")
        so_array = None if so is None else self._as_array(so, "so")
        sg_array = None if sg is None else self._as_array(sg, "sg")

        if sw_array is None:
            sw_array = 1.0 - so_array - sg_array
        elif so_array is None:
            so_array = 1.0 - sw_array - sg_array
        elif sg_array is None:
            sg_array = 1.0 - sw_array - so_array

        sw_array, so_array, sg_array = np.broadcast_arrays(
            sw_array,
            so_array,
            sg_array,
        )

        return sw_array, so_array, sg_array

    @staticmethod
    def _validate_saturations(
        sw: np.ndarray,
        so: np.ndarray,
        sg: np.ndarray,
        atol: float = 1e-8,
    ) -> None:
        """
        Validate phase saturations.

        Checks that:

            0 <= sw <= 1
            0 <= so <= 1
            0 <= sg <= 1
            sw + so + sg = 1

        Parameters
        ----------
        sw : numpy.ndarray
            Water saturation.
        so : numpy.ndarray
            Oil saturation.
        sg : numpy.ndarray
            Gas saturation.
        atol : float, default=1e-8
            Absolute tolerance for the saturation-sum check.
        """

        for name, sat in {"sw": sw, "so": so, "sg": sg}.items():
            if np.any((sat < 0.0) | (sat > 1.0)):
                raise ValueError(f"{name} must be between 0 and 1.")

        if not np.allclose(sw + so + sg, 1.0, atol=atol):
            raise ValueError("Saturations must sum to 1.")

    def get(
        self,
        sw: Optional[ArrayLike] = None,
        so: Optional[ArrayLike] = None,
        sg: Optional[ArrayLike] = None,
        check_sum: bool = True,
    ) -> ThreePhaseRelPerm:
        """
        Compute three-phase relative permeabilities.

        At least two of sw, so, and sg must be provided. The missing saturation
        is calculated from:

            sw + so + sg = 1

        Parameters
        ----------
        sw : float or array-like, optional
            Water saturation.
        so : float or array-like, optional
            Oil saturation.
        sg : float or array-like, optional
            Gas saturation.
        check_sum : bool, default=True
            If True, checks that sw + so + sg = 1.

        Returns
        -------
        ThreePhaseRelPerm
            Named tuple containing:

            krw : numpy.ndarray
                Three-phase water relative permeability.
            kro : numpy.ndarray
                Three-phase oil relative permeability.
            krg : numpy.ndarray
                Three-phase gas relative permeability.
        """

        sw_array, so_array, sg_array = self._prepare_saturations(sw, so, sg)

        if check_sum:
            self._validate_saturations(sw_array, so_array, sg_array)
        else:
            for name, sat in {"sw": sw_array, "so": so_array, "sg": sg_array}.items():
                if np.any((sat < 0.0) | (sat > 1.0)):
                    raise ValueError(f"{name} must be between 0 and 1.")

        krw, kro_ow = self.ow.get(sw_array)

        liquid_saturation = sw_array + so_array
        kro_go, krg = self.go.get(liquid_saturation)

        kro = self.kro(
            sw=sw_array,
            so=so_array,
            sg=sg_array,
            kro_ow=kro_ow,
            kro_go=kro_go,
        )

        # Use raw, unclipped oil normalized saturation for endpoint correction.
        sod_raw = self.sod(so_array, clip=False)

        kro = np.where(sod_raw < 0.0, 0.0, kro)
        kro = np.where(sod_raw > 1.0, self.kro0_ow, kro)

        kro = np.clip(kro, 0.0, self.kro0_ow)

        return ThreePhaseRelPerm(
            krw=krw,
            kro=kro,
            krg=krg,
        )

if __name__ == "__main__":

    rp = StoneModelI(swr=0.1,sor_ow=0.4,sor_go=0.2,sgr=0.05,krw0=0.3,kro0_ow=0.8,kro0_go=0.8,krg0=0.3)

    print(rp.get(0.3,0.5,0.2))

    rp = StoneModelI(swr=0.15,sor_ow=0.15,sor_go=0.05,sgr=0.1,krw0=0.3,kro0_ow=0.88,kro0_go=0.8,krg0=0.3)

    print(rp.kro(0.3,0.4,0.3,0.406,0.175))

    N = 20

    i, j = np.meshgrid(np.arange(N+1), np.arange(N+1))

    mask = i + j <= N

    i, j = i[mask], j[mask]

    sw = i / N
    so = j / N
    # sg = 1.0 - sw - so

    # print(sg)
    
    krw,kro,krg = rp.get(sw=sw,so=so)
