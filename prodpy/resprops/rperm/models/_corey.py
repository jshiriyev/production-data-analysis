import numpy as np

class Corey():
    """
    Implements the Corey-type relative permeability model for two-phase system.

    """

    def __init__(self,swr:float,snwr:float,korw:float,kornw:float):
        """Initializes the class.

        Parameters:
        ----------
        swr   : float
            residual saturation of wetting phase
        snwr  : float
            residual saturation of non-wetting phase.
        korw  : float
            End-point wetting phase relative permeability (at snwr).
        kornw : float
            End-point non-wetting phase relative permeability (at swr).
        lamda : float

        """
        self.swr  = swr
        self.snwr = snwr

        self.korw  = korw
        self.kornw = kornw

        self.lamda = lamda

    def swd(self,sw:np.ndarray):
        """Returns dimensionless saturation values."""
        return np.clip((np.ravel(sw)-self.swr)/(1-self.swr),0,1)

    def get(self,sw:np.ndarray):
        """Computes relative permeabilities at a given wetting phase saturation.

        Parameters:
        ----------
        sw   : numpy array
            wetting phase saturation.

        Returns:
        -------
        krw  : numpy array
            wetting phase relative permeability.
        krnw : numpy array
            non-wetting phase relative permeability.

        """
        S = self.swd(sw)

        krw = self.korw*S**(2/self.lamda+3)
        krnw = self.kornw*(1-S)**2*(1-S**(2/self.lamda+1))

        return krw,krnw

    @staticmethod
    def mobility(krw:np.ndarray,kro:np.ndarray,muw:np.ndarray,muo:np.ndarray):
        """Returns mobility values"""
        lambda_w = np.ravel(krw)/np.ravel(muw)  # Water mobility
        lambda_o = np.ravel(kro)/np.ravel(muo)  # Oil mobility

        return lambda_w/lambda_o