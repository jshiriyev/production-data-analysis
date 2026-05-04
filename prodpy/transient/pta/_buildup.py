from matplotlib import pyplot

import numpy

# from ._pressure import pressure

from ._items import well

class buildup():
    """Calculates well bottomhole pressure in field units for the given constant
    flow rate production and shutin period.
    """

    def __init__(self,_well):

        self._prodwell = well(radius=_well.radius,skin=_well.skin)
        self._injwell = well(radius=_well.radius,skin=_well.skin)

        self.rates = numpy.array(_well.rate)
        self.times = numpy.array(_well.time)

        self.totalprod = numpy.sum(self.rates*self.times)

        non_zero_rate_rates = self.rates[self.rates!=0]
        non_zero_rate_times = self.times[self.rates!=0]

        most_freq_rate_index = numpy.argmax(non_zero_rate_times/numpy.sum(non_zero_rate_times))

        self.qprod = non_zero_rate_rates[most_freq_rate_index]

        self.tprod = self.totalprod/self.qprod

        last_prod_index = numpy.nonzero(self.times)[0][-1]

        self.tshut = numpy.sum(self.times[last_prod_index:])

        self.ttotal = self.tprod+self.tshut

        self._prodwell.rate = self.qprod
        self._prodwell.time = self.ttotal

        self._injwell.rate = -self.qprod
        self._injwell.time = self.tshut

    def set_parameters(self,res,oil):

        self._pprod = pressure(res,oil,self._prodwell)

        self._pinj = pressure(res,oil,self._injwell)

    def set_irreducible(self,water,saturation):

        self._pprod.set_irreducible(water,saturation)

        self._pinj.set_irreducible(water,saturation)

    def set_compressibility(self,total=None):

        self._pprod.set_compressibility(total)

        self._pinj.set_compressibility(total)

    def initialize(self,scale="linear",size=50):

        twellprod = self._pprod.twell
        twellinj = self._pinj.twell

        if scale=="linear":
            tprods = numpy.linspace(twellprod,self.tprod,size)
            tshuts = numpy.linspace(twellinj,self.tshut,size)
        elif scale=="log":
            tprods = numpy.logspace(*numpy.log10([twellprod,self.tprod]),size)
            tshuts = numpy.logspace(*numpy.log10([twellinj,self.tshut]),size)

        ttotals = numpy.append(tprods,self.tprod+tshuts)

        self._pprod.initialize(time=ttotals)

        self._pinj.initialize(time=tshuts)

    def view(self,axis=None):

        showFlag = True if axis is None else False

        if axis is None:
            figure,axis = pyplot.subplots(nrows=1,ncols=1)

        yaxis,ylabel = -self.delta,f"Wellbore Pressure Change [psi]"

        axis.plot(self._pprod.time,yaxis)

        axis.set_xscale("linear")

        axis.set_ylim(ymin=0)

        axis.invert_yaxis()

        axis.set_xlabel("Time [days]")
        axis.set_ylabel(ylabel)

        if showFlag:
            pyplot.show()

    def horner(self,axis=None):

        showFlag = True if axis is None else False

        if axis is None:
            figure,axis = pyplot.subplots(nrows=1,ncols=1)

        dt = self.time-self.tprod#[1:]-self.time[:-1]

        delta = self.sdelta

        dp = delta-delta[0]

        ht = (self.tprod+dt[1:])/dt[1:]

        slope = (dp[1:]-dp[:-1])/(dt[1:]-dt[:-1])

        Pd = (slope[:-1]+slope[1:])/2*ht[:-1]*dt[1:-1]**2/self.qprod

        axis.plot(dt[1:],dp[1:])
        axis.plot(dt[1:-1],Pd)

        axis.set_xscale("log")
        axis.set_yscale("log")

        axis.set_xlabel("Delta Time [days]")
        axis.set_ylabel("Pressure Difference and Derivative")

        if showFlag:
            pyplot.show()

    @property
    def delta(self):

        delta = self._pprod.delta.copy()

        delta[self._pprod.time>self.tprod] += self._pinj.delta

        return delta

    @property
    def sdelta(self):

        return self.delta[self._pprod.time>=self.tprod]

    @property
    def time(self):

        return self._pprod.time[self._pprod.time>=self.tprod]
    

