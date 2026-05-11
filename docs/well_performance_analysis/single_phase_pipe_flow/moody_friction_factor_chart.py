import matplotlib.pyplot as plt

import numpy as np

from nodepy import pipes
from respy import Fluid

def darcy(diam,epd,rho):

	pipe = pipes.Pipe(diam,rough=epd*diam)

	fluid = Fluid(rho)

	dw = pipes.onephase.DarcyWeisbach(pipe,fluid)

	reynolds = np.logspace(2,8)

	rates = reynolds*np.pi*(fluid._visc*pipe.diam)/(4*fluid._rho)

	fD = dw.friction(rates)

	return reynolds,fD

r1,f1 = darcy(0.01,2e-2,1000)
r2,f2 = darcy(0.01,2e-3,1000)
r3,f3 = darcy(0.01,2e-4,1000)

fig, ax = plt.subplots(figsize=(10,6))

ax.loglog(r1,f1/4,linewidth=0.5,label='0.02')
ax.loglog(r2,f2/4,linewidth=0.5,label='0.002')
ax.loglog(r3,f3/4,linewidth=0.5,label='0.0002')

# ax.set_yscale('log')
# ax.set_xscale('log')

ax.set_xlim((600,1e8))
ax.set_ylim((0.002,0.025))

ax.legend()

# plt.tight_layout()

plt.show()