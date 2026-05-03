from matplotlib import pyplot

import numpy

from scipy import integrate

from scipy.optimize import root_scalar

from scipy.sparse import csr_matrix as csr
from scipy.sparse.linalg import spsolve as sps

from scipy.special import expi

from scipy.special import j0 as BJ0
from scipy.special import j1 as BJ1
from scipy.special import y0 as BY0
from scipy.special import y1 as BY1

class agarwal():
    """
    tD  : dimensionless time (k*t)/(phi*mu*ct*rw**2)
    CD  : dimensionless wellbore storage constant C/(2*pi*phi*h*ct*rw**2)
    SF  : skin factor, dimensionless
    """

    gamma = 0.57722

    @staticmethod
    def pressure(tD,CD=0,SF=0):
        """Returns the pressure solution to diffusivity equation for the well with
            - finite dimensions,
            - wellbore storage effect,
            - skin effect,
            - infinite reservoir
        Equation # 9, page 281
        """

        tD,CD,SF = numpy.array(tD).flatten(),float(CD),float(SF)

        if CD==0:
            return 1/2*(numpy.log(4*tD)-agarwal.gamma)+SF

        u = numpy.logspace(-8,1,2000)
        # u = numpy.linspace(1e-8,1e1,100000)

        if SF<0:
            TSF,SF = SF,0
            tD = tD*numpy.exp(2*TSF)
            CD = CD*numpy.exp(2*TSF)
            u = u/numpy.exp(2*TSF)

        pwD = numpy.zeros(tD.shape)

        for index,time in enumerate(tD):
            y = agarwal.pressure_integrand(u,time,CD,SF)
            z = integrate.simpson(y,u)

            pwD[index] = 4*z/numpy.pi**2

        return pwD

    @staticmethod
    def pressure_integrand(u,tD,CD=0,SF=0):
        """Integral kernel of the equation # 9, page 281"""

        tD,CD,SF = numpy.array(tD).flatten(),float(CD),float(SF)

        term1 = 1-numpy.exp(-u**2*tD)
        term2 = u*CD*BJ0(u)-(1-CD*SF*u**2)*BJ1(u)
        term3 = u*CD*BY0(u)-(1-CD*SF*u**2)*BY1(u)

        return term1/(u**3*(term2**2+term3**2))

    @staticmethod
    def pressure_lineSource(tD,CD=0,SF=0):
        """Equation # 11, page 281"""

        tD,CD,SF = numpy.array(tD).flatten(),float(CD),float(SF)

        u = numpy.logspace(-8,1,2000)

        if SF<0:
            TSF,SF = SF,0
            tD *= numpy.exp(2*TSF)
            CD *= numpy.exp(2*TSF)
            u /= numpy.exp(2*TSF)

        pwDline = numpy.zeros(tD.shape)

        for index,time in enumerate(tD):
            y = agarwal.pressure_lineSource_integrand(u,time,CD,SF)
            z = integrate.simpson(y,u)

            pwDline[index] = z

        return pwDline

    @staticmethod
    def pressure_lineSource_integrand(u,tD,CD=0,SF=0):
        """Internal kernel of the equation # 11, page 281"""

        tD,CD,SF = numpy.array(tD).flatten(),float(CD),float(SF)

        term1 = (1-numpy.exp(-u**2*tD))*BJ0(u)
        term2 = 1-u**2*CD*SF+1/2*numpy.pi*u**2*CD*BY0(u)
        term3 = 1/2*numpy.pi*CD*u**2*BJ0(u)

        return term1/(u*(term2**2+term3**2))

    @staticmethod
    def pressure_shorttime(tD,CD=0,SF=0):
        """Short-time approximation of pressure equation, Equation #14 & 15, page 282"""

        tD,CD,SF = numpy.array(tD).flatten(),float(CD),float(SF)

        if SF==0 and CD!=0:
            term1 = tD-(4*tD**(3/2))/(3*CD*numpy.sqrt(numpy.pi))

            return 1/CD*term1

        elif SF!=0 and CD!=0:
            term1 = tD-(tD**2)/(2*CD*SF)
            term2 = (8*tD**(5/2))/(15*numpy.sqrt(numpy.pi)*CD*SF**2)

            return 1/CD*(term1+term2)

        else:
            raise Warning("Not Implemented")

    @staticmethod
    def pressure_longtime(tD,CD=0,SF=0):
        """Long-time approximation of pressure equation, Equation #13, page 282"""

        tD,CD,SF = numpy.array(tD).flatten(),float(CD),float(SF)

        term1 = numpy.log(4*tD)-agarwal.gamma
        term2 = term1+2*SF

        return 1/2*(term2+1/(2*tD)*(term1+1-2*CD*term2))

    @staticmethod
    def derivative(tD,CD=0,SF=0):
        """Equation #26, page 284"""

        tD,CD,SF = numpy.array(tD).flatten(),float(CD),float(SF)

        pwDder = numpy.zeros(tD.shape)

        for index,time in enumerate(tD):
            u = numpy.logspace(-8,-1,1000)
            y = agarwal.derivative_integrand(u,time,CD,SF)
            z = integrate.simpson(y,u)

            pwDder[index] = 4*CD*z/numpy.pi**2

        return pwDder

    @staticmethod
    def derivative_integrand(u,tD,CD=0,SF=0):
        """Integral kernel of the equation 26, page 284"""

        tD,CD,SF = numpy.array(tD).flatten(),float(CD),float(SF)

        term1 = numpy.exp(-u**2*tD)
        term2 = u*CD*BJ0(u)-(1-CD*SF*u**2)*BJ1(u)
        term3 = u*CD*BY0(u)-(1-CD*SF*u**2)*BY1(u)

        return term1/(u*(term2**2+term3**2))

    @staticmethod
    def derivative_shorttime(tD,CD=0,SF=0):
        """Short-time approximation, equation 27 & 28, page 284"""

        tD,CD,SF = numpy.array(tD).flatten(),float(CD),float(SF)

        if SF==0 and CD!=0:
            return (1-2*numpy.sqrt(tD/numpy.pi)/CD+1/CD*(1/CD-1/2)*tD)/CD
        elif SF!=0 and CD!=0:
            return (1-tD/(CD*SF))

    @staticmethod
    def derivative_longtime(tD,CD=0,SF=0):
        """Long-time approximation, equation 29, page 284"""

        tD,CD,SF = numpy.array(tD).flatten(),float(CD),float(SF)

        term1 = CD/(2*tD)
        term2 = CD**2/(2*tD**2)*(2*SF-1)
        term3 = CD*(1-2*CD)/(4*tD**2)*(numpy.log(4*tD)-agarwal.gamma)

        return (term1+term2-term3)

if __name__ == "__main__":

    print(finite.setnodes(10,r=2.5))

    # tD = 1e-2

    # umin = numpy.sqrt(1/tD)/1e6
    # umax = numpy.sqrt(1/tD)*1e5

    # u = numpy.logspace(numpy.log10(umin),numpy.log10(umax),2000)

    # y1 = everdingen.pressure_integrand(u,tD)
    # y2 = everdingen.pressure_integrand(u,1e8)
    # y1 = everdingen.pressure_integrand(u,0.0001)

    # y1 = agarwal.pressure_integrand(u,1e8,1e5,0)
    # y2 = agarwal.pressure_integrand(u,1e8,1e5,0)
    # y2 = agarwal.pressure_lineSource_integrand(u,1e3,10000,0)
    # y3 = agarwal.pressure_lineSource_integrand(u,1e3,10000,5)
    # y4 = agarwal.pressure_lineSource_integrand(u,1e3,10000,10)
    # y5 = agarwal.pressure_lineSource_integrand(u,1e3,10000,20)

    # print(f"{agarwal.pressure(1e8,1e5,0)[0]:8.5f}")

    # pyplot.loglog(u,y1)
    # pyplot.loglog(u,y1)
    # pyplot.loglog(u,y2)
    # pyplot.loglog(u,y3)
    # pyplot.loglog(u,y4)
    # pyplot.loglog(u,y5)

    # pyplot.show()



