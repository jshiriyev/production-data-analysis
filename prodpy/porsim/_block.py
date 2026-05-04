import numpy as np

from ._cuboid import Cuboid
from ._vector import Vector

class Block(Cuboid):

    def __init__(self,grids,rrock,fluid,tcomp=None):
        """Initialization of block (cell) calculation class."""
        super().__init__(grids,rrock,fluid,tcomp)

    @property
    def mean(self):
        return Mean
    
    def vector(self,tstep=1.,wells=None,edges=None):
        """Returns vector to be used in the building."""
        avect = self.Avec(tstep) # accumulation vector
        tvect = self.Tvec()      # inter-block transmissibility vector
        wvect = self.Wvec(wells) # wells' block productivity vector list
        evect = self.Bvec(edges) # edges' block productivity vector list

        return Vector(avect,*tvect,wvect,evect)

    def Avec(self,_tstep:float):
        """Returns accumulation multiplied compressibility (A.ct).
        Input time-step, _tstep, must be in SI units."""
        avect = (self._volume*self.rrock._poro)/(_tstep)

        return avect*self._tcomp

    def Tvec(self):
        """Returns xflux, yflux, and zflux in tuple."""
        fluid = (self._mobil,self._power)

        xvect = self.mean.diffuse(self._xflow,*fluid,self._xneg,self._xpos)
        yvect = self.mean.diffuse(self._yflow,*fluid,self._yneg,self._ypos)
        zvect = self.mean.diffuse(self._zflow,*fluid,self._zneg,self._zpos)

        return xvect,yvect,zvect

    def Wvec(self,wells):
        """Returns productivity for all active wells."""
        prods,wells = [],() if wells is None else wells

        for well in wells:

            xsize = self._xdelta[list(well.index)]
            ysize = self._ydelta[list(well.index)]
            zsize = self._zdelta[list(well.index)]

            xperm = self.rrock._xperm[list(well.index)]
            yperm = self.rrock._yperm[list(well.index)]
            zperm = self.rrock._zperm[list(well.index)]

            if well.axis=="x":
                k1,k2,w1,w2,w3 = yperm,zperm,ysize,zsize,xsize
            elif well.axis=='y':
                k1,k2,w1,w2,w3 = zperm,xperm,zsize,xsize,ysize
            elif well.axis=='z':
                k1,k2,w1,w2,w3 = xperm,yperm,xsize,ysize,zsize

            wprop = (well._radius,well.skin)

            well._prod = self.mean.potency(k1,k2,w1,w2,w3,*wprop)*self._mobil[list(well.index)]
        
        return wells

    def Bvec(self,edges):
        """Returns transmissibility for all active boundaries."""
        prods,edges = [],() if edges is None else edges

        for edge in edges:

            bools = getattr(self,f"_{edge.face}")
            flows = getattr(self,f"_{edge.axis}flow")[bools]

            edge._prod = flows*self._mobil[bools]

        return edges

class Mean:

    @staticmethod
    def weighted(term1,term2):
        """Returns weighted mean of two terms, term1 and term 2."""
        return (term1+term2)/2

    @staticmethod
    def harmonic(term1,term2):
        """Returns harmonic mean of two terms, term1 and term 2."""
        return (2*term1*term2)/(term1+term2)

    @staticmethod
    def geometric(term1,term2):
        """Returns geometric mean of two terms, term1 and term 2."""
        return np.sqrt(term1*term2)

    @staticmethod
    def upwinded(term1,term2,ppot1,ppot2):
        """Computes the upwinded mean of two terms based on phase potential (ppot) values."""
        return np.where(ppot1<ppot2,term2,term1)

    @staticmethod
    def diffuse(flow_,mobil,power,_neg,_pos):
        """Helper function to compute mean harmonic and upwinded mobility product."""
        mean_condunce = Mean.harmonic(flow_[_neg],flow_[_pos])
        mean_mobility = Mean.upwinded(mobil[_neg],mobil[_pos],power[_neg],power[_pos])

        return mean_condunce*mean_mobility

    @staticmethod
    def radius(perm1,perm2,delta1,delta2):
        """Returns equivalent radius of grids that contains well."""
        sqrt21 = np.power(perm2/perm1,1/2)*np.power(delta1,2)
        sqrt12 = np.power(perm1/perm2,1/2)*np.power(delta2,2)
        quar21 = np.power(perm2/perm1,1/4)
        quar12 = np.power(perm1/perm2,1/4)

        return 0.28*np.sqrt(sqrt21+sqrt12)/(quar21+quar12)

    @staticmethod
    def potency(perm1,perm2,delta1,delta2,delta3,well_radius,well_skin):
        """Returns well's productivity value in a given grid."""
        mean_perm = Mean.geometric(perm1,perm2)
        mean_radi = Mean.radius(perm1,perm2,delta1,delta2)

        return (2*np.pi*delta3*mean_perm)/(np.log(mean_radi/well_radius)+well_skin)

if __name__ == "__main__":

    pass