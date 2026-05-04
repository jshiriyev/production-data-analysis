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

class everdingen():

    @staticmethod
    def pressure(tD):

        tD = numpy.array(tD).flatten()

        pwD = numpy.zeros(tD.shape)

        for index,time in enumerate(tD):

            umin = numpy.sqrt(1/time)/1e6
            umax = numpy.sqrt(1/time)*1e5

            u = numpy.logspace(numpy.log10(umin),numpy.log10(umax),2000)

            y = everdingen.pressure_integrand(u,time)
            z = integrate.simpson(y,u)

            pwD[index] = 4*z/numpy.pi**2

        return pwD

    @staticmethod
    def pressure_integrand(u,tD):
        """Integral kernel of the equation #VI-24, page 313"""

        tD = numpy.array(tD).flatten()

        term1 = 1-numpy.exp(-u**2*tD)
        term2 = BJ1(u)**2+BY1(u)**2

        return term1/(u**3*term2)

    @staticmethod
    def pressure_bounded(tD,R,numterms=2):

        tD,R = numpy.array(tD).flatten(),float(R)

        term1 = 2/(R**2-1)*(1/4.+tD)

        term2 = (3*R**4-4*R**4*numpy.log(R)-2*R**2-1)/(4*(R**2-1)**2)

        tD = tD.reshape((-1,1))

        betan = everdingen.pressure_bounded_find_roots(R,numterms)

        # print(f"{R=} {betan=}")

        betan = betan.reshape((1,-1))

        term3a = numpy.exp(-betan**2*tD)*BJ1(betan*R)**2

        term3b = betan**2*(BJ1(betan*R)**2-BJ1(betan)**2)

        term3 = (term3a/term3b).sum(axis=1)

        return term1-term2+2*term3

    @staticmethod
    def pressure_bounded_find_roots(R,numterms=2):

        roots = numpy.empty(numterms)

        for index in range(numterms):

            lower_bound = ((2*index+1)*numpy.pi)/(2*R-2)
            upper_bound = ((2*index+3)*numpy.pi)/(2*R-2)

            bracket = (lower_bound,upper_bound)

            solver = root_scalar(everdingen.pressure_bounded_root_function,
                args=(R,),method="brentq",bracket=bracket)

            roots[index] = solver.root

        return roots

    @staticmethod
    def pressure_bounded_root_function(beta,R):

        return BJ1(beta*R)*BY1(beta)-BJ1(beta)*BY1(beta*R)

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



