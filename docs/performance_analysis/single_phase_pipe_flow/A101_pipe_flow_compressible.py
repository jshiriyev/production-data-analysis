import matplotlib.pyplot as plt

import numpy as np

from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

def objcritratio(omega,friction_factor,length,diameter):

	lhs = 8*friction_factor*length/diameter

	rhs = (1/omega)**2-(np.log(1/omega))**2-1

	return (rhs-lhs)**2

def massflow(P1,P2,molw,temp,phi,length,diameter):

	UGC = 8.314     # universal gas constant

	csa = np.pi*diameter**2/4

	rho1 = (P1*molw)/(UGC*temp)
	rho2 = (P2*molw)/(UGC*temp)

	Dp = P1**2-P2**2
	Rp = np.log(P1/P2)
	Ft = 4*phi*length/diameter

	return csa*np.sqrt((rho1*Dp)/(2*P1*(Rp+Ft)))

def objdownspress(P2,P1,G,molw,temp,phi,length,diameter):

	UGC = 8.314     # universal gas constant

	rho1 = (P1*molw)/(UGC*temp)
	rho2 = (P2*molw)/(UGC*temp)

	Dp = P1**2-P2**2
	Rp = np.log(P1/P2)
	Ft = 4*phi*length/diameter

	Gc = csa*np.sqrt((rho1*Dp)/(2*P1*(Rp+Ft)))

	return (G-Gc)**2

T = 20+273.15	# Kelvin

L = 10_000		# meters
D = 12*0.0254	# meters

Mw = 0.016		# kilogram per moles
mu = 1.03e-5	# Pascal-second

G = 10 			# kilogram per seconds

Pup = 200*6894.76  # Pascal

csa = np.pi*D**2/4

Re = G*D/csa/mu

f = 0.0396/Re**(1/4)

Pdn = np.logspace(-5,np.log10(Pup),1000)

G = massflow(Pup,Pdn,Mw,T,f,L,D)

plt.semilogx(Pdn,G)
plt.show()

# omegac = minimize_scalar(objcritratio,args=(f,L,D),bounds=((1e-5,1)),method='bounded').x

# Pupc = omegac*Pup

# print("Downstream pressure is {} psi when mass rate is maximum".format(Pupc/6894.76))

# Pdown = minimize_scalar(objdownspress,args=(Pup,G,Mw,T,f,L,D),bounds=((Pupc,Pup)),method='bounded').x

# print("Downstream pressure is {} psi when mass rate is {} kg/sec".format(Pdown/6894.76,G))

# rho1 = (Pup*Mw)/(8.314*T)
# rho2 = (Pdown*Mw)/(8.314*T)

# u1 = G/(rho1*csa)
# u2 = G/(rho2*csa)
