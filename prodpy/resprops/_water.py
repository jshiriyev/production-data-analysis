import numpy as np

class WaterPhase():
    """
    The objective of this class is to present several of the well-established
    physical property correlations for the reservoir water:

    """

    def __init__(self):

        pass

    @staticmethod
    def visc(T:float|np.ndarray, p:float|np.ndarray=None, TDS:float|np.ndarray=0.):
        """Meehan (1980) proposed a water viscosity correlation that accounts
        for both the effects of pressure and salinity:
        
        Inputs:
        ------
        T    : Temperature of interest, °F
        p    : Pressure of interest, psia
        
        TDS  : Total dissolved solids, wt%, Y = 10000 * TDS, where Y is the
            water salinity in ppm.
        
        Brill and Beggs (1978) presented a simpler equation, which considers
        only temperature effects. If the pressure is None, the values are
        calculated based on this method.

        Returns:
        -------
        Calculated viscosity in cp.

        """
        if p is None:
            return np.exp(1.003 - 1.479e-2*T + 1.982e-5*T**2)

        Y = 10000 * TDS

        A = 0.04518 + 9.313e-7 * Y - 3.93e-12 * Y ** 2
        B = 70.634 + 9.576e-10 * Y ** 2

        # brine viscosity at p = 14.7, T, cp
        muwd = A + B / T

        mu = muwd * (1 + 3.5e-2 * p ** 2 * (T - 40))

        return mu

    @staticmethod
    def rho(Bw, TDS):
        """Function to Calculate Water Density in lb/ft
        
        Bw   water formation volume factor, bbl/stb
        TDS  total dissolved solids, wt%

        """
        return (62.368 + 0.438603 * TDS + 1.60074e-3 * TDS ** 2) / Bw

    @staticmethod
    def salinity(spgr):
        """Function to Calculate Water Salinity at 60°F and 1 atm

        spgr specific gravity of water

        """
        rho = 62.368 * spgr

        a = 0.00160074
        b = 0.438603
        c = 62.368 - rho
        s = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)

        return s

    @staticmethod
    def comp(T:float|np.ndarray, p:float|np.ndarray):
        """Brill and Beggs (1978) proposed the following equation for estimating
        water isothermal compressibility, ignoring the corrections for dissolved
        gas and solids:

        Inputs:
        ------
        T    : Temperature of interest, °F
        p    : Pressure of interest, psia

        Returns:
        -------
        Cw   : Water compressibility in psi-1

        """
        C1 = 3.8546 - 0.000134 * p
        C2 = -0.01052 + 4.77e-7 * p
        C3 = 3.9267e-5 - 8.8e-10 * p

        return (C1 + C2*T + C3*T**2) * 1e-6

    @staticmethod
    def fvf(T:float|np.ndarray, p:float|np.ndarray, TDS:float|np.ndarray=0., option:str="gas-free-water"):
        """Function to Calculate Water Formation Volume Factor in bbl/stb
        
        Inputs:
        ------
        T      : Temperature of interest, °F
        p      : Pressure of interest, psia

        TDS    : Total dissolved solids, wt%, Y = 10000 * TDS, where Y is the
            water salinity in ppm.

        option : water type, "gas-free-water" or "gas-saturated-water"

        Returns:
        -------
        Water formation volume factor (Bw) in bbl/STB

        """
        match option:

            case "gas-free-water":
                C1 = 0.9947 + 5.8e-6 * T + 1.02e-6 * T ** 2
                C2 = -4.228e-6 + 1.8376e-8 * T - 6.77e-11 * T ** 2
                C3 = 1.3e-10 - 1.3855e-12 * T + 4.285e-15 * T ** 2
            case "gas-saturated-water":
                C1 = 0.9911 + 6.35e-5 * T + 8.5e-7 * T ** 2
                C2 = -1.093e-6 - 3.497e-9 * T + 4.57e-12 * T ** 2
                C3 = -5e-11 + 6.429e-13 * T - 1.43e-15 * T ** 2
            case _:
                raise ValueError("Invalid option. Choose 'gas-free-water' or 'gas-saturated-water'.")  

        Bw = C1 + C2 * p + C3 * p ** 2

        Y = 10000 * TDS
        x = 5.1e-8 * p + (T - 60) * (5.47e-6 - 1.95e-10 * p) + (T - 60) ** 2 * (-3.23e-8 + 8.5e-13 * p)

        Bw = Bw * (1 + 0.0001 * x * Y)

        return Bw

    @staticmethod
    def gass(T:float|np.ndarray, p:float|np.ndarray, TDS:float|np.ndarray=0.):
        """The following correlation can be used to determine the gas solubility
        in water in scf/stb:
        
        Inputs:
        ------
        T      : Temperature of interest, °F
        p      : Pressure of interest, psia

        TDS    : Total dissolved solids, wt%, Y = 10000 * TDS, where Y is the
            water salinity in ppm.

        Returns:
        -------
        Gas solubility in water in scf/stb.

        """
        C1 = 2.12 + 3.45e-3 * T - 3.59e-5 * T ** 2
        C2 = 1.07e-2 - 5.26e-5 * T + 1.48e-07 * T ** 2
        C3 = 8.75e-7 + 3.90e-9 * T - 1.02e-11 * T ** 2

        Rswp = C1 + C2 * p + C3 * p ** 2

        Y = 10000 * TDS
        x = 3.471 * T ** -0.837

        Rsw = Rswp * (1 - 0.0001 * x * Y)

        return Rsw

    @staticmethod
    def tens(T:float|np.ndarray, p:float|np.ndarray):
        """Function to Calculate Gas-Water Interfacial Tension in dynes/cm
        
        Inputs:
        ------
        T      : Temperature of interest, °F
        p      : Pressure of interest, psia
        
        """
        s74 = 75 - 1.108 * p ** 0.349
        s280 = 53 - 0.1048 * p ** 0.637
        if (T <= 74):
            sw = s74
        elif(T >= 280):
            sw = s280
        else:
            sw = s74 - (T - 74) * (s74 - s280) / 206
        
        if (sw < 1):
            sw = 1
        
        return sw