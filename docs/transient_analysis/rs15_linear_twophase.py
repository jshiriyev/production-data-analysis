import unittest

import numpy as np

import _setup

from flow.pormed.linear import singlephase
from flow.pormed.linear import multicomponent
from flow.pormed.linear import multiphase

Sw = 0.3
Sg = 0.3

RP = relative_permeability(Sw,Sg,Som=0.1125,Sorow=0.15,Sorgo=0.05,Swc=0.15,Sgc=0.1,krowc=0.88)

RP.system3phase(model="Stone's Model II")

print(RP.kro,RP.krw,RP.krg)

k = 50 #mD
k *= 0.986923e-15

phi = 0.15
mu = 1e-3
c = 2e-9

sp = diffusivity(k/(phi*mu*c))

PL = 2000 #psi
PU = 3500 #psi

PL *= 6894.76
PU *= 6894.76

sp.cartesian_1D((0,50),(PL,PU),np.array([60,600]),x=np.array([12,27,41]),N=4)

print(sp.pressure/6894.76)

beta = 10

bound_points = np.array([[0],[7]])

bound_conditions = np.array([[1,0,300],[0,1,0]])

analytical = slightly_compressible(0)

analytical.cartesian_poisson_1D(bound_points,bound_conditions,beta)

plt.plot(analytical.x,analytical.u,'k',label='Analytical Solution')

plt.tight_layout()

plt.show()

## ALL CALCULATIONS ARE CARRIED IN BUCKLEYLEVERETT CLASS ABOVE
## BASED ON THE INPUT PROVIDED BELOW

Sor = 0.25
Swi = 0.15
Swr = 0.15
##    koro = 1.0
##    korw = 0.78
##    m = 2.6
##    n = 3.7
muo = 8.
muw = 1.

q = 1000*5.615
A = 2500
L = 1000
phi = 0.2

BL = buckleyleverett(Sor,Swr,muo,muw)

##    BL.coreymodel(koro,korw,m,n)
BL.k_model()
BL.fractionalflow()
BL.shockfront(Swi)

BL.production(q,A,L,phi)
BL.profile(q,A,phi,BL.tbt*0.3)#0.02 is for the time

Np = BL.Np*BL.Vp/5.615          # oil produced, bbl

qo = np.empty_like(Np)
qo[0] = Np[1]/BL.tp[1]
qo[1:] = Np[1:]/BL.tp[1:]       # bbl/day

qw = q/5.615-qo                 # bbl/day

WOR = qw/qo

WOR[WOR<1e-15] = 0

# INPUTS END HERE

# THE SECTION BELOW IS ONLY FOR PLOTTING AND ANIMATION
# FOR SHOWING A PLOT YOU WANT, UNCOMMENT OUT THE LINES ACCORDINGLY

# RELATIVE PERMEABILITY PLOT

plt.plot(BL.Sw,BL.kro)
plt.plot(BL.Sw,BL.krw)
plt.xlim([0,1])
plt.ylim(bottom=0)
plt.xlabel('water saturation',fontsize=14)
plt.ylabel('relative permeability',fontsize=14)
plt.legend(('oil','water'),fontsize=14)
plt.show()

# FRACTIONAL FLOW PLOT

plt.plot(BL.Sw,BL.fw,c='k')
plt.xlabel('water saturation',fontsize=14)
plt.ylabel('fractional water flow',fontsize=14)
plt.xlim((0,1))
plt.ylim((0,1))
##    plt.xticks([0,1])
##    plt.yticks([0,1])
plt.show()

# FRACTIONAL FLOW DERIVATIVE PLOT

##    A = (BL.Sw-BL.Swr)**3*BL.muo
##    B = A+2*(1-BL.Sw-BL.Sor)**2*BL.muw
##
##    C = 3*(BL.Sw-BL.Swr)**2*BL.muo
##    D = C-4*(1-BL.Sw-BL.Sor)*BL.muw
##
##    F = C/B-A/B**2*D

plt.plot(BL.Sw,BL.fw_der,c='k')
##    plt.plot(BL.Sw,F,'r--')
plt.xlabel('water saturation',fontsize=14)
plt.ylabel('derivative of water fractional flow',fontsize=14)
plt.xlim((0,1))
plt.ylim(bottom=0)
##    plt.xticks([0,1])
##    plt.yticks([])
plt.show()

# FRACTIONAL FLOW AND ITS DERIVATIVE TOGETHER PLOT

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.plot(BL.Sw,BL.fw,color=color)
ax1.set_ylabel('water fractional flow',color=color)
ax1.set_xlabel('water saturation')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:red'
ax2.plot(BL.Sw,BL.fw_der,color=color)
ax2.set_ylabel('water fractional flow derivative',color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()

plt.xlim((0,1))

plt.show()

# FRACTIONAL FLOW PLOT FOR THE SHOCK FRONT DETERMINATION

plt.plot(BL.Sw,BL.fw,c='k')
plt.plot((BL.Swi,BL.Sw_avg),(BL.fwi,1),c='r')
plt.xlabel('water saturation',fontsize=14)
plt.ylabel('fractional water flow',fontsize=14)
plt.xlim((0,1))
plt.ylim((0,1))
##    plt.xticks([0,1])
##    plt.yticks([0,1])
plt.show()

# WATER SATURATION PORFILE WITH RESPECT TO DISTANCE

x = np.insert(BL.x_Sw,0,2*L)
y = np.insert(BL.Sw_IC,0,BL.Swi)
plt.plot(x,y,'k')
plt.xlim((0,L))
plt.ylim((0,1))
plt.xlabel('x-direction',fontsize=14)
plt.ylabel('water saturation',fontsize=14)
plt.tight_layout()
plt.show()

# OIL PRODUCTION RATE VS INJECTED WATER

plt.plot(BL.Wi,BL.Np*BL.Vp/5.615/BL.tp)
plt.xlim([0,3])
plt.ylim(bottom=0)
plt.xlabel('Pore Volume of Water injected',fontsize=14)
plt.ylabel('Oil Production Rate [bbl/day]',fontsize=14)
plt.tight_layout()
plt.grid()
plt.show()

# WATER OIL RATIO VS CUMULATIVE OIL PRODUCTION

plt.semilogx(WOR,Np)
plt.ylim(bottom=90000)
plt.xlim(left=0.00001)
plt.xlabel('water oil ratio',fontsize=14)
plt.ylabel('total oil production [bbl]',fontsize=14)
plt.tight_layout()
plt.grid()
plt.show()

# CUMULATIVE OIL RECOVERY VS INJECTED WATER

plt.plot(BL.Wi,BL.Np)
plt.xlim([0,10])
plt.ylim(bottom=0)
plt.xlabel('Injected Pore Volume of Water',fontsize=14)
plt.ylabel('Produced Pore Volume of Oil',fontsize=14)
plt.tight_layout()
plt.grid()
plt.show()

# ANIMATION OF WATER SATURATION WITH RESPECT TO DISTANCE CHANGING WITH TIME WHILE SWEEPING OIL

ax = plt.subplot(1,1,1)

ax.set_xlim((0,L))
ax.set_ylim((0,1))
ax.set_xlabel('x-direction',fontsize=14)
ax.set_ylabel('water saturation',fontsize=14)

plt.tight_layout()

N_tn = 100

t_n = np.linspace(0,0.035,N_tn)

frames = []

for i,t in enumerate(t_n):
   
   BL.profile(q,A,phi,t)
   
   x = np.insert(BL.x_Sw,0,2*L)
   y = np.insert(BL.Sw_IC,0,BL.Swi)

   line, = ax.plot(x,y,'k')

   #below the image is created in the memory and appended to frames
   buf = io.BytesIO()
   plt.savefig(buf)
   img = Image.open(buf)
   frames.append(img)

   line.remove()
   
plt.close()

#generates the animation from the images created
frames[0].save('buckley.gif',
              format='GIF',
              append_images=frames[1:],
              save_all=True,
              duration=50,
              loop=0)
                   
if __name__ == "__main__":

    unittest.main()
