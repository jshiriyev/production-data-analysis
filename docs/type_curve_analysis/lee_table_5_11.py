import matplotlib.pyplot as plt

import numpy as np

data = np.loadtxt("lee_table_5_11.txt",skiprows=1)

t,pwf,pseudo = data.T

plt.scatter(t,pseudo,s=3,color='black')

plt.xscale("log")

plt.xlabel('Flowing Time [hours]')
plt.ylabel(r'Pseudo Flowing Pressure $[psi^2/cp]$')

plt.grid()

plt.show()