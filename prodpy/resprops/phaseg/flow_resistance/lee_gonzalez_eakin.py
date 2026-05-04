import numpy

def gvisc(P, T, Z, grav):
    """Function to Calculate Gas Viscosity in cp"""
    #P          pressure, psia
    #T          temperature, °R
    #Z          gas compressibility factor
    #grav       gas specific gravity
    M = 28.964 * grav
    x = 3.448 + 986.4 / T + 0.01009 * M
    Y = 2.447 - 0.2224 * x
    rho = (1.4926 / 1000) * P * M / Z / T
    if Y<0 or rho<0:
       print ('epa')
    K = (9.379 + 0.01607 * M) * T ** 1.5 / (209.2 + 19.26 * M + T)
    return K * math.exp(x * rho ** Y) / 10000

class LeeGonzalezEakin():
    """Lee-Gonzalez-Eakin Method"""
    
    def __init__(self,spgr,temp):
        """Viscosity class that can be used for ResPy, returning viscosities
            for pressures when called. Initialization parameters are:
        
        spgr    : specific gravity of the gas at standard conditions.

        temp    : temperature value (Rankine) which will be used to calculate
                  viscosities when the class is called.
        """

        self._spgr = spgr
        self._molw = self.get_molw(spgr)/1000 # Molecular Mass, kg/mol

        self._temp = temp*(5./9)

        self.A = self.get_A()
        self.B = self.get_B()
        self.C = self.get_C()

    @property
    def spgr(self):
        """specific gravity of the gas at standard conditions"""
        return self._spgr
    
    @property
    def molw(self):
        """Molecular Mass, lb/lbmol"""
        return self._molw*1000

    @property
    def temp(self):
        return self._temp*(9./5)

    def get_A(self):
        upper = (9.379+0.01607*self.molw)*self.temp**1.5
        lower = (209.2+19.26*self.molw+self.temp)
        return upper/lower

    def get_B(self):
        return 3.448+986.4/self.temp+0.01009*self.molw

    def get_C(self):
        return 2.447-0.2224*self.B
    
    def __call__(self,press,zfact):
        """Returns Gas Viscosity in cp"""

        rho = (press*self.molw)/(zfact*10.731577089016*self.temp)
        
        return self.A*numpy.exp(self.B*rho**self.C)/10000

    @staticmethod
    def get_molw(spgr):
        return spgr*28.964

    @staticmethod
    def get_spgr(molw):
        return molw/28.964

if __name__ == "__main__":

    a = LGE(0.8,200)

    print(a._T)

    print(a(500,0.9))

    def L(x):
        return x**2

    c = 2

    print(callable(c))