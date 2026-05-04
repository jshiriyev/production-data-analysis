import numpy as np

class Fluid():
    """
    A class that defines constant fluid properties at the
    given pressure and temperature.
    """

    def __init__(self,visc,*,rho=62.4,comp=1e-6,fvf=1.0,press=None,satur=1.0,rperm=1.0):
        """
        Initializes a fluid with specific properties.

        Parameters
        ----------
        visc    : float or array-like of floats, optional
            Viscosity of the fluid in centipoise (cp)

        rho     : float or array-like of floats, optional
            Density of the fluid in lb/ft3

        comp    : float or array-like of floats, optional
            Compressibility of the fluid in 1/psi.

        fvf     : float or array-like of floats, optional
            Formation volume factor in ft3/scf

        press   : float or array-like of floats, optional
            Pressure at which properties are defined in psi
        
        satur   : float or array-like of floats, optional
             Saturation value (dimensionless).

        rperm   : float or array-like of floats, optional
            Relative permeability value (dimensionless).

        """
        self.visc  = visc
        self.rho   = rho
        
        self.comp  = comp
        self.fvf   = fvf

        self.press = press
        self.satur = satur
        self.rperm = rperm

        self.grad  = None
        self.mobil = None

    @property
    def visc(self):
        """Getter for the fluid viscosity."""
        return self._visc/0.001

    @visc.setter
    def visc(self,value):
        """Setter for the fluid viscosity."""
        self._visc = np.ravel(value).astype(float)*0.001

    @property
    def rho(self):
        """Getter for the fluid density."""
        return self._rho/16.0185

    @rho.setter
    def rho(self,value):
        """Setter for the fluid density."""
        self._rho = np.ravel(value).astype(float)*16.0185

    @property
    def comp(self):
        """Getter for the fluid compressibility."""
        return self._comp*6894.76

    @comp.setter
    def comp(self,value):
        """Setter for the fluid compressibility."""
        self._comp = np.ravel(value).astype(float)/6894.76

    @property
    def fvf(self):
        """Getter for the fluid formation volume factor."""
        return self._fvf

    @fvf.setter
    def fvf(self,value):
        """Setter for the fluid formation volume factor."""
        self._fvf = np.ravel(value).astype(float)

    @property
    def press(self):
        """Getter for the fluid pressure."""
        return None if self._press is None else self._press/6894.76

    @press.setter
    def press(self,value):
        """Setter for the fluid pressure."""
        self._press = None if value is None else np.ravel(value).astype(float)*6894.76

    @property
    def satur(self):
        """Getter for the fluid saturation."""
        return self._satur

    @satur.setter
    def satur(self,value):
        """Setter for the fluid saturation."""
        self._satur = np.ravel(value).astype(float)

    @property
    def rperm(self):
        """Getter for the fluid relative permeability."""
        return self._rperm

    @rperm.setter
    def rperm(self,value):
        """Setter for the fluid relative permeability."""
        self._rperm = np.ravel(value).astype(float)

    @property
    def grad(self):
        """Getter for the fluid gradient."""
        return self._grad/22620.6

    @grad.setter
    def grad(self,value):
        """Setter for the fluid gradient."""
        self._grad = self._rho*9.807

    @property
    def mobil(self):
        """Getter for the fluid mobility."""
        return self._mobil*0.001

    @mobil.setter
    def mobil(self,value):
        """Setter for the fluid mobility."""
        self._mobil = (self._rperm)/(self._visc*self._fvf)

if __name__ == "__main__":

    f = Fluid(visc=0.5,rho=62.4,fvf=1)

    print(f)

    print(f.visc,f._visc)
    print(f.rho,f._rho)
    print(f.grad,f._grad)

    print(f.comp)
    print(f.mobil)


    