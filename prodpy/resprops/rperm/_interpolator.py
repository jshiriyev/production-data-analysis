from typing import Union, Literal

import numpy as np
import pandas as pd

from scipy import interpolate

ArrayLike = Union[list[float], np.ndarray, pd.Series]

class Interpolator:
    """
    Interpolates two-phase relative permeability data.

    The class accepts either a pandas DataFrame or separate array-like inputs
    for saturation, wetting-phase relative permeability, and oil/non-wetting
    relative permeability.

    Parameters
    ----------
    data : pandas.DataFrame, optional
        DataFrame containing saturation and relative permeability data.
    Sw_array : array-like, optional
        Known wetting-phase saturation values.
    krw_array : array-like, optional
        Known wetting-phase relative permeability values.
    kro_array : array-like, optional
        Known oil/non-wetting-phase relative permeability values.
    Sw_col : str, default="Sw"
        Column name for wetting-phase saturation if `data` is provided.
    krw_col : str, default="krw"
        Column name for wetting-phase relative permeability if `data` is provided.
    kro_col : str, default="kro"
        Column name for oil/non-wetting-phase relative permeability if `data` is provided.

    method : {"spline", "pchip"}, default="pchip"
        Interpolation method. ``pchip`` is usually safer for relative permeability
        because it reduces non-physical oscillations.
    spline_order : int, default=3
        Degree of the spline. Use 1 for linear, 2 for quadratic, 3 for cubic.
    smoothing : float, default=0
        Smoothing factor. Use 0 to interpolate exactly through the data points.
        This option is used only when ``method="spline"``.
    clip : bool, default=True
        If True, clip interpolated or extrapolated relative permeability values
        to the interval [0, 1]. Saturation inputs are not clipped.

    Notes
    -----
    Input data are sorted internally by saturation. Duplicate saturation values
    are rejected. Both interpolation methods extrapolate outside the provided
    saturation range; use ``clip=True`` to keep extrapolated relative
    permeability values inside physical bounds.
    
    """

    def __init__(
        self,
        *args,
        data: pd.DataFrame | None = None,
        Sw_col: str = "Sw",
        krw_col: str = "krw",
        kro_col: str = "kro",
        Sw_array: ArrayLike | None = None,
        krw_array: ArrayLike | None = None,
        kro_array: ArrayLike | None = None,
        method: Literal["spline", "pchip"] = "pchip",
        spline_order: int = 3,
        smoothing: float = 0.0,
        clip: bool = True,
    ):
        
        if len(args) == 1:
            if data is not None:
                raise ValueError(
                    "Ambiguous input. Do not provide both a positional input "
                    "and keyword input such as `data`."
                )
            if isinstance(args[0], pd.DataFrame):
                data = args[0]
            else:
                raise ValueError(
                    "Invalid positional argument. Expected a pandas DataFrame. "
                    "For all other inputs, please use keyword arguments."
                )
        elif len(args) > 1:
            raise ValueError(
                "Too many positional arguments were provided. Expected at most one "
                "positional argument, such as a pandas DataFrame. For all other inputs, "
                "please use keyword arguments."
            )

        self.method = method
        self.spline_order = spline_order
        self.smoothing = smoothing
        self.clip = clip

        if data is not None:
            self._load_from_dataframe(
                data=data,
                Sw_col=Sw_col,
                krw_col=krw_col,
                kro_col=kro_col,
            )
        else:
            self._load_from_arrays(
                Sw_array=Sw_array,
                krw_array=krw_array,
                kro_array=kro_array,
            )

        self._validate_array_shapes()
        self._prepare_data()
        self._validate_inputs()
        self._fit()

    def _load_from_dataframe(
        self,
        data: pd.DataFrame,
        Sw_col: str,
        krw_col: str,
        kro_col: str,
    ):
        """Load interpolation data from a pandas DataFrame."""

        required_columns = [Sw_col, krw_col, kro_col]

        missing_columns = [
            col for col in required_columns if col not in data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns in DataFrame: {missing_columns}"
            )

        df = data[required_columns].copy()
        df = df.dropna()
        df = df.sort_values(by=Sw_col)

        self.Sw_array = df[Sw_col].to_numpy(dtype=float)
        self.krw_array = df[krw_col].to_numpy(dtype=float)
        self.kro_array = df[kro_col].to_numpy(dtype=float)

    def _load_from_arrays(
        self,
        Sw_array: ArrayLike | None,
        krw_array: ArrayLike | None,
        kro_array: ArrayLike | None,
    ) -> None:
        """Load interpolation data from separate array-like inputs."""

        if Sw_array is None or krw_array is None or kro_array is None:
            raise ValueError(
                "Provide either a DataFrame using `data`, or provide "
                "`Sw_array`, `krw_array`, and `kro_array`."
            )

        self.Sw_array = np.asarray(Sw_array, dtype=float)
        self.krw_array = np.asarray(krw_array, dtype=float)
        self.kro_array = np.asarray(kro_array, dtype=float)

    def _prepare_data(self) -> None:
        """Sort data by saturation."""

        order = np.argsort(self.Sw_array)

        self.Sw_array = self.Sw_array[order]
        self.krw_array = self.krw_array[order]
        self.kro_array = self.kro_array[order]

    def _validate_array_shapes(self) -> None:
        """Validate array dimensions before sorting."""

        if self.Sw_array.ndim != 1:
            raise ValueError("Sw_array must be one-dimensional.")

        if self.krw_array.ndim != 1 or self.kro_array.ndim != 1:
            raise ValueError("krw_array and kro_array must be one-dimensional.")

    def _validate_inputs(self):
        """Validate input data before interpolation."""

        if self.method not in {"spline", "pchip"}:
            raise ValueError("method must be either 'spline' or 'pchip'.")

        if not (
            len(self.Sw_array) == len(self.krw_array) == len(self.kro_array)
        ):
            raise ValueError(
                "Sw_array, krw_array, and kro_array must have the same length."
            )
        
        if len(self.Sw_array) < 2:
            raise ValueError("At least two data points are required.")
        
        if not (
            np.all(np.isfinite(self.Sw_array))
            and np.all(np.isfinite(self.krw_array))
            and np.all(np.isfinite(self.kro_array))
        ):
            raise ValueError(
                "Input data must contain only finite numeric values."
            )
        
        if np.any(np.diff(self.Sw_array) <= 0):
            raise ValueError(
                "Sw_array must be strictly increasing after sorting. "
                "Check for duplicate saturation values."
            )
        
        if self.method == "spline":
            if not 1 <= self.spline_order <= 5:
                raise ValueError("spline_order must be between 1 and 5.")

            if len(self.Sw_array) < self.spline_order + 1:
                raise ValueError(
                    f"At least {self.spline_order + 1} data points are required "
                    f"for spline_order={self.spline_order}."
                )

    def _fit(self):
        """Fit spline functions for krw and kro."""

        if self.method == "pchip":
            self._krw_interp = interpolate.PchipInterpolator(
                self.Sw_array,
                self.krw_array,
                extrapolate=True,
            )

            self._kro_interp = interpolate.PchipInterpolator(
                self.Sw_array,
                self.kro_array,
                extrapolate=True,
            )

        else:
            self._krw_interp = interpolate.splrep(
                self.Sw_array,
                self.krw_array,
                s=self.smoothing,
                k=self.spline_order,
            )

            self._kro_interp = interpolate.splrep(
                self.Sw_array,
                self.kro_array,
                s=self.smoothing,
                k=self.spline_order,
            )

    def get(self, Sw_new):
        """
        Return interpolated krw and kro values at new saturation values.

        Parameters
        ----------
        Sw_new : float or array-like
            Saturation value or values where relative permeability
            should be interpolated.

        Returns
        -------
        krw_new : float or ndarray
            Interpolated wetting-phase relative permeability.
        kro_new : float or ndarray
            Interpolated oil/non-wetting-phase relative permeability.
        """
        scalar_input = np.isscalar(Sw_new)
        Sw_new = np.asarray(Sw_new, dtype=float)

        if self.method == "pchip":
            krw_new = self._krw_interp(Sw_new)
            kro_new = self._kro_interp(Sw_new)
        else:
            krw_new = interpolate.splev(Sw_new, self._krw_interp, der=0)
            kro_new = interpolate.splev(Sw_new, self._kro_interp, der=0)

        if self.clip:
            krw_new = np.clip(krw_new, 0.0, 1.0)
            kro_new = np.clip(kro_new, 0.0, 1.0)
        
        if scalar_input:
            return float(krw_new), float(kro_new)

        return krw_new, kro_new
    
if __name__ == "__main__":

    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        {
            "Sw": [0.25, 0.30, 0.40, 0.50, 0.60, 0.65],
            "Krw": [0.000, 0.018, 0.092, 0.198, 0.327, 0.400],
            "Kro": [0.850, 0.754, 0.557, 0.352, 0.131, 0.000],
        }
    )

    relperm = Interpolator(
        data=df,
        Sw_col="Sw",
        krw_col="Krw",
        kro_col="Kro",
    )

    krw, kro = relperm.get(0.45)

    print(f"Interpolated krw at Sw=0.45: {krw:.4f}")
    print(f"Interpolated kro at Sw=0.45: {kro:.4f}")

    Sw_array = [0.25, 0.30, 0.40, 0.50, 0.60, 0.65]
    krw_array = [0.00, 0.018, 0.092, 0.198, 0.327, 0.40]
    kro_array = [0.85, 0.754, 0.557, 0.352, 0.131, 0.00]

    relperm = Interpolator(Sw_array=Sw_array, krw_array=krw_array, kro_array=kro_array)

    Sw_new = np.linspace(0.25, 0.65, 100)

    krw_new, kro_new = relperm.get(Sw_new)

    plt.plot(Sw_array, krw_array, 'ro', label='Data (krw)')
    plt.plot(Sw_array, kro_array, 'bo', label='Data (kro)')
    plt.plot(Sw_new, krw_new, 'r-', label='Spline (krw)')
    plt.plot(Sw_new, kro_new, 'b-', label='Spline (kro)')
    
    plt.show()
