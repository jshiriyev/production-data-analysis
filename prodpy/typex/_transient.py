from matplotlib import pyplot

import numpy

from scipy.special import expi

from ._typecurve import TypeCurve

class LineSource(TypeCurve):
    """Calculates well bottomhole pressure in field units for the given
    constant flow rate oil production, single phase flow."""

    def __init__(self,*args,**kwargs):
        """Initializes reservoir, oil and wcond properties."""

        super().__init__(*args,**kwargs)

    def press(self):
        """It calculates delta bottom hole flowing pressure for constant
        flow rate production at transient state, exponential integral solution."""

        pseudo_cond = self.time>=self.tbound

        pseudo_time = self.time[pseudo_cond]

    def transient(self,times):

        Ei = expi(-39.5*(self.wcond.radius**2)/(self.diffuse*times))

        return self.pzero-self.pcons*(1/2*Ei-self.wcond.skin)

    def pseudosteady(self,times):
        """
        Sets the radius of outer circular boundary, ft and calculates
        pseudo steady state solution correcting pressure for boundary effects.
        
        If area is not None, the radius is calculated from area [acre].
        """

        term_well = -0.012648*(self.diffuse*times)/self.wcond.radius**2

        term_edge = -numpy.log(self.rrock.radius/self.wcond.radius)

        return self.pzero-self.pcons*(term_well+term_edge+3/4-self.wcond.skin)

    @property
    def twell(self):
        return (self.wcond.radius**2)/(self.diffuse)*15802

    @property
    def tedge(self):
        return (self.rrock.radius**2)/(self.diffuse)*39.5

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



