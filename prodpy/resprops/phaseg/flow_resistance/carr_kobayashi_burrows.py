import numpy

class CarrKobayashiBurrows():
    """Carr-Kobayashi-Burrows Correlation Method"""

    b0 = -2.46211820,2.970547414,-2.86264054e-1,8.05420522e-3
    b1 =  2.80860949,-3.49803305, 3.60373020e-1,-1.044324e-2
    b2 = -7.93385648e-1,1.39643306,-1.49144925e-1,4.41015512e-3
    b3 =  8.39387178e-2,-1.86408848e-1,2.03367881e-2,-6.09579263e-4

    def __init__(self,spgr,temp,crit,yN2=0,yCO2=0,yH2S=0):
        """Viscosity class that can be used for ResPy, returning viscosities
            for pressures when called. Initialization parameters are:
        
        spgr    : specific gravity of the gas at standard conditions.

        temp    : temperature value (Rankine) which will be used to calculate
                  viscosities when the class is called.

        crit    : tuple of (pcrit in psi, tcrit in Rankine)

        yN2     : mole fraction of nitrogen (N2).
        yCO2    : mole fraction of carbon dioxide (CO2).
        yH2S    : mole fraction of hydrogen sulfide (H2S)
        """

        self._spgr = spgr

        self._temp = temp*(5./9)

        pcrit,tcrit = crit

        self._pcrit = pcrit*6894.76
        self._tcrit = tcrit*(5./9)

        mu0 = self.uncorrected(spgr,temp)

        muN = self.N2corr(spgr,yN2)
        muC = self.CO2corr(spgr,yCO2)
        muS = self.H2Scorr(spgr,yH2S)

        self.mu1 = self.corrected(mu0,muN,muC,muS)

    @property
    def spgr(self):
        return self._spgr

    @property
    def temp(self):
        return self._temp*(9./5)

    @property
    def pcrit(self):
        """Critical Pressure in psi"""
        return self._pcrit/6894.76

    @property
    def tcrit(self):
        """Critical Temperature in Rankine"""
        return self._tcrit*(9./5)

    @property
    def tred(self):
        """Returns reduced temperature (class property)."""
        return self.temp/self.tcrit

    def __call__(self,press):

        return self.mu1*self.ratio(self.pred(press),self.tred)

    def pred(self,press):
        """Returns reduced pressure values for input pressure values in psi."""
        return press/self.pcrit

    @staticmethod
    def ratio(self,pred,tred):
        """Dempsey (1965) viscosity ratio, mu/mu1
        
        pred    : reduced pressure
        tred    : reduced temperature
        """

        A0 = CKB.polynomial(*CKB.b0,pred)
        A1 = CKB.polynomial(*CKB.b1,pred)
        A2 = CKB.polynomial(*CKB.b2,pred)
        A3 = CKB.polynomial(*CKB.b3,pred)

        poly = CKB.polynomial(A0,A1,A2,A3,tred)

        return numpy.exp(poly)/tred

    @staticmethod
    def polynomial(c0,c1,c2,c3,x):
        return c0+c1*x+c2*x**2+c3*x**3

    @staticmethod
    def corrected(mu0,dmuN2,dmuCO2,dmuH2S):
        """Corrected gas viscosity at one atmospheric
        pressure and reservoir temperature, cp"""
        return mu0+dmuN2+dmuCO2+dmuH2S

    @staticmethod
    def uncorrected(spgr,temp):
        """Uncorrected gas viscosity, cp

        temp   : Temperature in Fahrenheits
        """
        return 8.188e-3-6.15e-3*numpy.log(spgr)\
            +(1.709e-5-2.062e-6*spgr)*temp

    @staticmethod
    def N2corr(spgr,y):
        """Visocsity correction due to the presence of N2:
        y : mole fraction of N2 in the gas."""
        return y*(8.48e-3*numpy.log(spgr)+9.59e-3)

    @staticmethod
    def CO2corr(spgr,y):
        """Visocsity correction due to the presence of CO2:
        y : mole fraction of CO2 in the gas"""
        return y*(9.08e-3*numpy.log(spgr)+6.24e-3)

    @staticmethod
    def H2Scorr(spgr,y):
        """Visocsity correction due to the presence of H2S:
        y : mole fraction of H2S in the gas."""
        return y*(8.49e-3*numpy.log(spgr)+3.73e-3)

if __name__ == "__main__":

    a = LGE(0.8,200)

    print(a._T)

    print(a(500,0.9))

    def L(x):
        return x**2

    c = 2

    print(callable(c))