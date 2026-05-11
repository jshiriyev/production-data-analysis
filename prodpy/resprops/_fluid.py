import numpy as np

from ._layer import Layer

class Fluid():
    """
    Fluid properties with oilfield-unit public access and SI-unit internal storage.
    It defines constant fluid properties at the given pressure and temperature.

    Public properties use oilfield units:
    visc : cp
    rho : lb/ft3
    comp : 1/psi
    press : psi
    grad : psi/ft
    mobil : 1/cp, computed as rperm / (visc * fvf)

    Internal underscored properties use SI units:
    _visc : Pa*s
    _rho : kg/m3
    _comp : 1/Pa
    _press : Pa
    _grad : Pa/m
    _mobil : 1/(Pa*s)

    NaN values are allowed to represent unknown grid/state values.
    """
    CP_TO_PA_S = 1e-3
    LBFT3_TO_KGM3 = 16.018463
    PSI_TO_PA = 6894.757293168
    PA_PER_M_TO_PSI_PER_FT = 22620.594
    _GRAVITY = 9.80665

    def __init__(self,visc,*,rho=62.4,comp=1e-6,fvf=1.0,press=None,satur=1.0,rperm=1.0,size:int|None=None):
        """
        Initializes a fluid with specific properties.

        Parameters
        ----------
        visc    : float or array-like of floats, positional
            Viscosity of the fluid in centipoise (cp)

        rho     : float or array-like of floats, optional
            Density of the fluid in lb/ft3

        comp    : float or array-like of floats, optional
            Compressibility of the fluid in 1/psi.

        fvf     : float or array-like of floats, optional
            Formation volume factor, dimensionless;
            bbl/STB for oil & water and ft3/scf for gasses.

        press   : float or array-like of floats, optional
            Pressure at which properties are defined in psi
        
        satur   : float or array-like of floats, optional
             Saturation value (dimensionless).

        rperm   : float or array-like of floats, optional
            Relative permeability value (dimensionless).

        size   : int, optional
            Expected size of array-like properties. If provided,
            all array-like properties must match this size.

        """
        self.size  = size

        self.visc  = visc        
        self.rho   = rho
        self.comp  = comp
        self.fvf   = fvf

        self.press = press
        self.satur = satur
        self.rperm = rperm

    def __repr__(self):
        """Return a concise constructor-style representation of the fluid."""

        args = [
            f"visc={Layer._repr_value(self.visc)}",
            f"rho={Layer._repr_value(self.rho)}",
            f"comp={Layer._repr_value(self.comp)}",
            f"fvf={Layer._repr_value(self.fvf)}",
            f"press={Layer._repr_value(self.press)}",
            f"satur={Layer._repr_value(self.satur)}",
            f"rperm={Layer._repr_value(self.rperm)}",
        ]

        return f"{type(self).__name__}({', '.join(args)})"

    @property
    def visc(self):
        """Viscosity in cp."""
        return self._visc/self.CP_TO_PA_S

    @visc.setter
    def visc(self,value):
        """Viscosity in Pa.s."""
        self._visc = Layer._validate_range(
            value, 0, size=self.size, name="visc", include_lower_limit = False,
            )*self.CP_TO_PA_S

    @property
    def rho(self):
        """Getter for the fluid density."""
        return self._rho/self.LBFT3_TO_KGM3

    @rho.setter
    def rho(self,value):
        """Setter for the fluid density."""
        self._rho = Layer._validate_range(
            value, 0, size=self.size, name="rho", include_lower_limit = False,
            )*self.LBFT3_TO_KGM3

    @property
    def comp(self):
        """Getter for the fluid compressibility."""
        return self._comp*self.PSI_TO_PA

    @comp.setter
    def comp(self,value):
        """Setter for the fluid compressibility."""
        self._comp = Layer._validate_range(value, 0, size=self.size, name="comp")/self.PSI_TO_PA

    @property
    def fvf(self):
        """Getter for the fluid formation volume factor."""
        return self._fvf

    @fvf.setter
    def fvf(self,value):
        """Setter for the fluid formation volume factor."""
        self._fvf = Layer._validate_range(
            value, 0, size=self.size, name="fvf", include_lower_limit = False,
            )

    @property
    def press(self):
        """Getter for the fluid pressure."""
        return None if self._press is None else self._press/self.PSI_TO_PA

    @press.setter
    def press(self,value):
        """Setter for the fluid pressure."""
        self._press = None if value is None else (
            Layer._validate_range(value, 0, size=self.size, name="press")*self.PSI_TO_PA
        )

    @property
    def satur(self):
        """Getter for the fluid saturation."""
        return self._satur

    @satur.setter
    def satur(self,value):
        """Setter for the fluid saturation."""
        self._satur = Layer._validate_range(value, 0, 1, size=self.size, name="satur")

    @property
    def rperm(self):
        """Getter for the fluid relative permeability."""
        return self._rperm

    @rperm.setter
    def rperm(self,value):
        """Setter for the fluid relative permeability."""
        self._rperm = Layer._validate_range(value, 0, 1, size=self.size, name="rperm")

    @property
    def grad(self):
        """Pressure gradient in psi/ft."""
        return self._grad / self.PA_PER_M_TO_PSI_PER_FT

    @property
    def _grad(self):
        """Pressure gradient in Pa/m."""
        return self._rho * self._GRAVITY

    @property
    def mobil(self):
        """Fluid mobility kr / (mu * B) in 1/cp."""
        return self._mobil * self.CP_TO_PA_S

    @property
    def _mobil(self):
        """Fluid mobility kr / (mu * B) in 1/(Pa*s)."""
        return (self._rperm)/(self._visc*self._fvf)

if __name__ == "__main__":

    f = Fluid(visc=0.5,rho=62.4,fvf=1)

    print(f)

    print(f.visc,f._visc)
    print(f.rho,f._rho)
    print(f.grad,f._grad)

    print(f.comp)
    print(f.mobil)