import numpy as np

##        DATA
##
##        Time (hour)	C/C0
##        0.35	0.075
##        0.37	0.215
##        0.385	0.37
##        0.396	0.5
##        0.41	0.65
##        0.43	0.83
##        0.44	0.89
##        0.46	0.96

# Section below should be used to test multicomponent model

# data = np.loadtxt('Lecture_08_dispersion',skiprows=1)

# time = data[:,0]*3600   # seconds
# cc0 = data[:,1]         # dimensionless

# D = 6*1e-2              # meter
# L = 40*1e-2             # meter

# phi = 0.35              # dimensionless

# q = 1000*1e-6/3600      # meter3/seconds

# A = np.pi*D**2/4
# v = q/A/phi

# T = optimize.minimize(obj,0,args=(v,L),method="Powell",bounds=[(0,10)])
# ##T = optimize.minimize_scalar(obj,args=(v,L),method="bounded",bounds=(0,10))

# ##DL = np.logspace(-10,-1,10000)
# ##
# ##OF = np.empty_like(DL)
# ##
# ##for i in range(10000):
# ##    OF[i] = obj(DL[i],v,L)
# ##
# ##plt.loglog(DL,OF)
# ##plt.show()

# ##DL = 4.140*1e-7         # meter2/second

# t = np.linspace(1100,1700)

# Cr = concentration(T.x,v,L,t)

# plt.scatter(time/3600,cc0)
# plt.xlabel('time in hours')
# plt.ylabel('concentration ratio')
# plt.plot(t/3600,Cr,'r')
# plt.show()
