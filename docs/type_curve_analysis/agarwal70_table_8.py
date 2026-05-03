import numpy

from typx.wtest import agarwal1970

tD = [.1,.2,.5,1,2,5,10,20,50,100,200,500,1_000,2_000,5_000,10_000,20_000,50_000,100_000]
tD = numpy.array(tD)*1000

pwD1 = agarwal1970.pressure_lineSource(tD,1e2,20)
pwD2 = agarwal1970.pressure_lineSource(tD,1e3,20)
pwD3 = agarwal1970.pressure_lineSource(tD,1e4,20)
pwD4 = agarwal1970.pressure_lineSource(tD,1e5,20)

for time,p1,p2,p3,p4 in zip(tD,pwD1,pwD2,pwD3,pwD4):
    print(f"{time:11,.0f} {p1:8.5f} {p2:8.5f} {p3:8.5f} {p4:8.5f}")