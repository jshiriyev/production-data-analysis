from matplotlib import pyplot

import numpy

from typx.wtest import agarwal1970

tD = numpy.logspace(2,7,100)

# st = agarwal1970.pressure_shorttime(tD,0,-5)
# lt = agarwal1970.pressure_longtime(tD,0,5)

ft01 = agarwal1970.pressure(tD,0,-5)
ft02 = agarwal1970.pressure(tD,100,-5)
ft03 = agarwal1970.pressure(tD,1000,-5)
ft04 = agarwal1970.pressure(tD,10000,-5)
ft05 = agarwal1970.pressure(tD,100000,-5)

# pl01 = agarwal1970.pressure_lineSource(tD,0,-5)
# pl02 = agarwal1970.pressure_lineSource(tD,100,-5)
# pl03 = agarwal1970.pressure_lineSource(tD,1000,-5)
# pl04 = agarwal1970.pressure_lineSource(tD,10000,-5)
# pl05 = agarwal1970.pressure_lineSource(tD,100000,-5)

# pd01 = agarwal1970.derivative(tD,0,-5)
# pd02 = agarwal1970.derivative(tD,100,-5)
# pd03 = agarwal1970.derivative(tD,1000,-5)
# pd04 = agarwal1970.derivative(tD,10000,-5)
# pd05 = agarwal1970.derivative(tD,100000,-5)

ft06 = agarwal1970.pressure(tD,0,0)
ft07 = agarwal1970.pressure(tD,100,0)
ft08 = agarwal1970.pressure(tD,1000,0)
ft09 = agarwal1970.pressure(tD,10000,0)
ft10 = agarwal1970.pressure(tD,100000,0)

# pl06 = agarwal1970.pressure_lineSource(tD,0,0)
# pl07 = agarwal1970.pressure_lineSource(tD,100,0)
# pl08 = agarwal1970.pressure_lineSource(tD,1000,0)
# pl09 = agarwal1970.pressure_lineSource(tD,10000,0)
# pl10 = agarwal1970.pressure_lineSource(tD,100000,0)

# pd06 = agarwal1970.derivative(tD,0,0)
# pd07 = agarwal1970.derivative(tD,100,0)
# pd08 = agarwal1970.derivative(tD,1000,0)
# pd09 = agarwal1970.derivative(tD,10000,0)
# pd10 = agarwal1970.derivative(tD,100000,0)

ft11 = agarwal1970.pressure(tD,0,5)
ft12 = agarwal1970.pressure(tD,100,5)
ft13 = agarwal1970.pressure(tD,1000,5)
ft14 = agarwal1970.pressure(tD,10000,5)
ft15 = agarwal1970.pressure(tD,100000,5)

# pl11 = agarwal1970.pressure_lineSource(tD,0,5)
# pl12 = agarwal1970.pressure_lineSource(tD,100,5)
# pl13 = agarwal1970.pressure_lineSource(tD,1000,5)
# pl14 = agarwal1970.pressure_lineSource(tD,10000,5)
# pl15 = agarwal1970.pressure_lineSource(tD,100000,5)

# pd11 = agarwal1970.derivative(tD,0,5)
# pd12 = agarwal1970.derivative(tD,100,5)
# pd13 = agarwal1970.derivative(tD,1000,5)
# pd14 = agarwal1970.derivative(tD,10000,5)
# pd15 = agarwal1970.derivative(tD,100000,5)

ft16 = agarwal1970.pressure(tD,0,10)
ft17 = agarwal1970.pressure(tD,100,10)
ft18 = agarwal1970.pressure(tD,1000,10)
ft19 = agarwal1970.pressure(tD,10000,10)
ft20 = agarwal1970.pressure(tD,100000,10)

# pl16 = agarwal1970.pressure_lineSource(tD,0,10)
# pl17 = agarwal1970.pressure_lineSource(tD,100,10)
# pl18 = agarwal1970.pressure_lineSource(tD,1000,10)
# pl19 = agarwal1970.pressure_lineSource(tD,10000,10)
# pl20 = agarwal1970.pressure_lineSource(tD,100000,10)

# pd16 = agarwal1970.derivative(tD,0,10)
# pd17 = agarwal1970.derivative(tD,100,10)
# pd18 = agarwal1970.derivative(tD,1000,10)
# pd19 = agarwal1970.derivative(tD,10000,10)
# pd20 = agarwal1970.derivative(tD,100000,10)

ft21 = agarwal1970.pressure(tD,0,20)
ft22 = agarwal1970.pressure(tD,100,20)
ft23 = agarwal1970.pressure(tD,1000,20)
ft24 = agarwal1970.pressure(tD,10000,20)
ft25 = agarwal1970.pressure(tD,100000,20)

# pl21 = agarwal1970.pressure_lineSource(tD,0,20)
# pl22 = agarwal1970.pressure_lineSource(tD,100,20)
# pl23 = agarwal1970.pressure_lineSource(tD,1000,20)
# pl24 = agarwal1970.pressure_lineSource(tD,10000,20)
# pl25 = agarwal1970.pressure_lineSource(tD,100000,20)

# pd21 = agarwal1970.derivative(tD,0,20)
# pd22 = agarwal1970.derivative(tD,100,20)
# pd23 = agarwal1970.derivative(tD,1000,20)
# pd24 = agarwal1970.derivative(tD,10000,20)
# pd25 = agarwal1970.derivative(tD,100000,20)

figure,axis = pyplot.subplots(nrows=1,ncols=1)

axis.plot(tD,ft01,color="k",linewidth=0.5,label="skin = -5")
axis.plot(tD,ft02,color="k",linewidth=0.5)
axis.plot(tD,ft03,color="k",linewidth=0.5)
axis.plot(tD,ft04,color="k",linewidth=0.5)
axis.plot(tD,ft05,color="k",linewidth=0.5)

# axis.plot(tD,pl01,color="r",linewidth=0.5,linestyle="--")
# axis.plot(tD,pl02,color="r",linewidth=0.5,linestyle="--")
# axis.plot(tD,pl03,color="r",linewidth=0.5,linestyle="--")
# axis.plot(tD,pl04,color="r",linewidth=0.5,linestyle="--")
# axis.plot(tD,pl05,color="r",linewidth=0.5,linestyle="--")

# axis.plot(tD,pd01,color="b",linewidth=0.5,linestyle='--')
# axis.plot(tD,pd02,color="b",linewidth=0.5,linestyle='--')
# axis.plot(tD,pd03,color="b",linewidth=0.5,linestyle='--')
# axis.plot(tD,pd04,color="b",linewidth=0.5,linestyle='--')
# axis.plot(tD,pd05,color="b",linewidth=0.5,linestyle='--')

axis.plot(tD,ft06,color="r",linewidth=0.5,label="skin = 0")
axis.plot(tD,ft07,color="r",linewidth=0.5)
axis.plot(tD,ft08,color="r",linewidth=0.5)
axis.plot(tD,ft09,color="r",linewidth=0.5)
axis.plot(tD,ft10,color="r",linewidth=0.5)

# axis.plot(tD,pl06,color="r",linewidth=0.5,linestyle="--")
# axis.plot(tD,pl07,color="r",linewidth=0.5,linestyle="--")
# axis.plot(tD,pl08,color="r",linewidth=0.5,linestyle="--")
# axis.plot(tD,pl09,color="r",linewidth=0.5,linestyle="--")
# axis.plot(tD,pl10,color="r",linewidth=0.5,linestyle="--")

# axis.plot(tD,pd06,color="b",linewidth=0.5,linestyle='--')
# axis.plot(tD,pd07,color="b",linewidth=0.5,linestyle='--')
# axis.plot(tD,pd08,color="b",linewidth=0.5,linestyle='--')
# axis.plot(tD,pd09,color="b",linewidth=0.5,linestyle='--')
# axis.plot(tD,pd10,color="b",linewidth=0.5,linestyle='--')

axis.plot(tD,ft11,color="b",linewidth=0.5,label="skin = 5")
axis.plot(tD,ft12,color="b",linewidth=0.5)
axis.plot(tD,ft13,color="b",linewidth=0.5)
axis.plot(tD,ft14,color="b",linewidth=0.5)
axis.plot(tD,ft15,color="b",linewidth=0.5)

# axis.plot(tD,pl11,color="r",linewidth=0.5,linestyle="--")
# axis.plot(tD,pl12,color="r",linewidth=0.5,linestyle="--")
# axis.plot(tD,pl13,color="r",linewidth=0.5,linestyle="--")
# axis.plot(tD,pl14,color="r",linewidth=0.5,linestyle="--")
# axis.plot(tD,pl15,color="r",linewidth=0.5,linestyle="--")

# axis.plot(tD,pd11,color="b",linewidth=0.5,linestyle='--')
# axis.plot(tD,pd12,color="b",linewidth=0.5,linestyle='--')
# axis.plot(tD,pd13,color="b",linewidth=0.5,linestyle='--')
# axis.plot(tD,pd14,color="b",linewidth=0.5,linestyle='--')
# axis.plot(tD,pd15,color="b",linewidth=0.5,linestyle='--')

axis.plot(tD,ft16,color="g",linewidth=0.5,label="skin = 10")
axis.plot(tD,ft17,color="g",linewidth=0.5)
axis.plot(tD,ft18,color="g",linewidth=0.5)
axis.plot(tD,ft19,color="g",linewidth=0.5)
axis.plot(tD,ft20,color="g",linewidth=0.5)

# axis.plot(tD,pl16,color="r",linewidth=0.5,linestyle="--")
# axis.plot(tD,pl17,color="r",linewidth=0.5,linestyle="--")
# axis.plot(tD,pl18,color="r",linewidth=0.5,linestyle="--")
# axis.plot(tD,pl19,color="r",linewidth=0.5,linestyle="--")
# axis.plot(tD,pl20,color="r",linewidth=0.5,linestyle="--")

# axis.plot(tD,pd16,color="b",linewidth=0.5,linestyle='--')
# axis.plot(tD,pd17,color="b",linewidth=0.5,linestyle='--')
# axis.plot(tD,pd18,color="b",linewidth=0.5,linestyle='--')
# axis.plot(tD,pd19,color="b",linewidth=0.5,linestyle='--')
# axis.plot(tD,pd20,color="b",linewidth=0.5,linestyle='--')

axis.plot(tD,ft21,color="m",linewidth=0.5,label="skin = 20")
axis.plot(tD,ft22,color="m",linewidth=0.5)
axis.plot(tD,ft23,color="m",linewidth=0.5)
axis.plot(tD,ft24,color="m",linewidth=0.5)
axis.plot(tD,ft25,color="m",linewidth=0.5)

# axis.plot(tD,pl21,color="r",linewidth=0.5,linestyle="--")
# axis.plot(tD,pl22,color="r",linewidth=0.5,linestyle="--")
# axis.plot(tD,pl23,color="r",linewidth=0.5,linestyle="--")
# axis.plot(tD,pl24,color="r",linewidth=0.5,linestyle="--")
# axis.plot(tD,pl25,color="r",linewidth=0.5,linestyle="--")

# axis.plot(tD,pd21,color="m",linewidth=0.5,linestyle='--')
# axis.plot(tD,pd22,color="m",linewidth=0.5,linestyle='--')
# axis.plot(tD,pd23,color="m",linewidth=0.5,linestyle='--')
# axis.plot(tD,pd24,color="m",linewidth=0.5,linestyle='--')
# axis.plot(tD,pd25,color="m",linewidth=0.5,linestyle='--')

axis.set_ylim((0.1,100))
axis.set_xlim((1e2,1e7))

axis.set_xscale("log")
axis.set_yscale("log")

axis.set_xlabel("Dimensionless Time, ${t_D}$")
axis.set_ylabel("Dimensionless Pressure, ${p_{wD}}$")

axis.legend()

axis.grid()

# axis.legend()

pyplot.show()