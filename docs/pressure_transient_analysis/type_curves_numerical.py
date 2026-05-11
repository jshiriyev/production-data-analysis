import numpy

import matplotlib.pyplot as plt

from typx.wtest import agarwal
from typx.wtest import everdingen
from typx.wtest import finite

from typx.scomp.finite import derive

# print(finite.gridinner(4.48,5,5/4))
# print(finite.gridouter(4.48,4,5/4))

tD = numpy.logspace(2,7,2000)
CD = 1000

pwD = agarwal.pressure(tD,CD,0)
pwDD = derive(pwD,tD)*tD

pwD_800 = everdingen.pressure_bounded(tD=tD,R=800,numterms=10)
pwDD_800 = derive(pwD_800,tD)*tD

sol = finite(800,10,0.3)
print(sol.skin)

tD2 = numpy.logspace(0,7,2000)

pwD_800num = sol.pressure(tD2,CD)
pwDD_800num = derive(pwD_800num,tD2)*tD2

line1 = plt.loglog(tD/CD,pwD,label="R -> inf, CD = 1000")[0]
plt.loglog(tD/CD,pwDD,color=line1.get_color())

line3 = plt.loglog(tD/CD,pwD_800,label="R = 800, CD = 0")[0]
plt.loglog(tD/CD,pwDD_800,color=line3.get_color())

line2 = plt.loglog(tD2[500:]/CD,pwD_800num[500:],color='k',label="Finite Difference")[0]
plt.loglog(tD2[500:]/CD,pwDD_800num[500:],color=line2.get_color())

plt.xlabel("${t_D/C_D}$")

plt.ylabel("${p_D}$     ${(t_D/C_D)p_D'}$")

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()
