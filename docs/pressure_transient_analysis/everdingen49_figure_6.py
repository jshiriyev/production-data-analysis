import numpy

import matplotlib.pyplot as plt

from typx.wtest import everdingen1949

tD = numpy.linspace(10,40,1000)

pwD_06 = everdingen1949.pressure_bounded(tD=tD,R=6,numterms=10)
pwD_08 = everdingen1949.pressure_bounded(tD=tD,R=8,numterms=10)
pwD_10 = everdingen1949.pressure_bounded(tD=tD,R=10,numterms=10)

pwD = everdingen1949.pressure(tD)

plt.plot(tD,pwD_06,label="R=6")
plt.plot(tD,pwD_08,label="R=8")
plt.plot(tD,pwD_10,label="R=10")
plt.plot(tD,pwD,label="R->infinite")

# plt.gca().invert_yaxis()

plt.xlim((10,40))
plt.ylim((2.3,1.8))

plt.legend()

plt.grid()

plt.show()