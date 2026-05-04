import numpy

from scipy.special import erfc

class OnePhase():

    def __init__(self,length:float,k:float,phi:float,mu:float,ct:float):
        """
        length  : length of the porous media, ft

        k       : permeability, milli-Darcy
        phi     : porosity, non-dimensional
        mu      : viscosity, centi-poise
        ct      : total compressibility, 1/psi

        """
        self.length = length*0.3048

        self.eta = (k*9.869233e-16)/(phi*(ct/6894.76)*(mu*1e-3))

    def solve(self,time:float,ngrids:int,pinit:float,pleft:float,pright=None,noflux=False):
        """
        time    : when to calculate pressures, in hours
        ngrids  : number of pressure calculation points
        pinit   : initial pressure, psi
        pleft   : upstream pressure, psi
        pright  : downstream pressure, psi
        noflux  : downstream pressure boundary conditions
                  True, no-flux boundary condition is implementted
                  False, constant pressure boundary condition is implemented

        returns x (ft) locations where pressures are calculated and pressure values (psi).

        """

        time *= 3600

        pinit *= 6894.76

        pleft *= 6894.76

        xaxis = numpy.arange(self.length/ngrids/2,self.length,self.length/ngrids)
        
        X = self.x2dim(xaxis,self.length)

        T = self.time2dim(time,self.eta,self.length)
        
        if not noflux: # constant downstream pressure case

            pright = 101325 if pright is None else pright*6894.76

            ndimPresInit = self.press2dim(pinit,pleft,pright)
            ndimPressure = self.solve_ndimpressure_pconst(X,T,ndimPresInit)

            dimPressure = self.dim2press(ndimPressure,pleft,pright)

            return xaxis/0.3048,dimPressure/6894.76

        ndimPressure = self.solve_ndimpressure_noflux(X,T)
        dimPressure = self.dim2press(ndimPressure,pleft,pinit)

        return xaxis/0.3048,dimPressure/6894.76

    @staticmethod
    def solve_ndimpressure_pconst(ndimx:numpy.ndarray,ndimtime:float,ndimPressInit:float,terms:int=100):
        """Calculates the non-dimensional pressure for constant pressure
        boundary conditions on both sides.

        ndimx           : non-dimensional distance
        ndimtime        : non-dimensional time
        ndimPressInit   : non-dimensional pressure at initial pressure conditions
        terms           : number of terms after which to terminate the summation

        """

        n,psum1,psum2 = 1,numpy.zeros(ndimx.shape),numpy.zeros(ndimx.shape)

        while n<=terms:

            sinterm1 = numpy.sin(n*numpy.pi*ndimx)
            sinterm2 = numpy.sin(n*numpy.pi*(ndimx-1))

            expterm = numpy.exp(-n**2*numpy.pi**2*ndimtime)
            
            psum1 += (-1)**n/n*sinterm1*expterm
            psum2 += (-1)**n/n*sinterm2*expterm

            n += 1

        return 1-ndimx-ndimPressInit*2/numpy.pi*psum1-(1-ndimPressInit)*2/numpy.pi*psum2

    @staticmethod
    def solve_ndimpressure_noflux(ndimx:numpy.ndarray,ndimtime:float,terms:int=100):
        """Calculates the non-dimensional pressure for constant pressure at upstream
        and no-flux at downstream.

        ndimx      : non-dimensional x
        ndimtime   : non-dimensional time
        terms      : number of terms after which to terminate the summation

        """

        n,psum = 0,numpy.zeros(ndimx.shape)

        while n<terms:
            
            expterm = numpy.exp(-(2*n+1)**2/4*numpy.pi**2*ndimtime)
            sinterm = numpy.sin((2*n+1)*numpy.pi/2*ndimx)
            
            # psum += (-1)**n/(2*n+1)*expterm*costerm
            psum += 1/(2*n+1)*expterm*sinterm
            # psum += 4/((2*n+1)*numpy.pi)*expterm*costerm

            n += 1

        return 1-4/numpy.pi*psum
    
    @staticmethod
    def x2dim(dimx,length):
        """Converts dimensional x to non-dimensional x."""
        return dimx/length

    @staticmethod
    def dim2x(ndimx,length):
        """Converts non-dimensional x to dimensional x."""
        return ndimx*length

    @staticmethod
    def time2dim(dimtime,eta,length):
        """Converts dimensional time to non-dimensional time."""
        return eta*dimtime/length**2

    @staticmethod
    def dim2time(ndimtime,eta,length):
        """Converts non-dimensional time to dimensional time."""
        return ndimtime*length**2/eta

    @staticmethod
    def press2dim(dimpressure,pb1,pb2):
        """Converts dimensional pressure to non-dimensional pressure."""
        return (dimpressure-pb2)/(pb1-pb2)

    @staticmethod
    def dim2press(ndimpressure,pb1,pb2):
        """Converts non-dimensional pressure to dimensional pressure."""
        return ndimpressure*(pb1-pb2)+pb2