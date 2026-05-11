import matplotlib.pyplot as plt

import numpy as np

data = np.loadtxt("lee_table_2_8.txt",skiprows=1)

dt,Pws = data[:,0],data[:,1] # hours, psi

Qo = 12_173  # STB

phi = 0.14

mu = 0.55 	 # cp
ct = 16e-6   # 1/psi
rw = 0.5     # ft

Awb = 0.0218 # ft2
re = 1320    # ft
rhoo = 54.8  # lbm/ft3
q = 988 	 # STB/day

B = 1.126 	 # bbl/STB
h = 7        # ft

A = np.pi*re**2

tp = Qo/q*24 # hours

dtt = (tp+dt[1:])/dt[1:]

dte = dt/(1+dt/tp)

PwsPwf = Pws-709

plt.figure()

plt.scatter(dtt,Pws[1:],s=3,c='k')

plt.grid(which='both')

plt.xscale("log")

plt.xlabel(r"$\frac{t_p+\Delta t}{\Delta t}$",fontsize=14)
plt.ylabel(r"$p_{ws}$",fontsize=14)

plt.xlim((200,3))
plt.ylim((2000,5000))

m = 550

x1 = np.linspace(3,200)
y1 = -m*np.log10(x1)+4850

k = 162.6*q*B*mu/m/h

print(k)

pws1 = -m*np.log10((tp+1))+4850

# print(tp,pws1)

# s = 1.151*((pws1-709)/m+np.log10(1688*phi*mu*ct*rw**2/k)+np.log10((tp+1)/tp))
s = 1.151*((pws1-709)/m-np.log10(k/(phi*mu*ct*rw**2))+3.23)

# print(k,s)

print("Ps = ",162.6*(q*B*mu)/(k*h)*0.0637)
print(0.000264*k*tp/(phi*mu*ct*A))

# plt.plot(x1,y1,color='red')

plt.tight_layout()

plt.figure()

plt.scatter(dte,PwsPwf,s=3,c='k')

plt.grid(which='both')

plt.xlabel(r"$\frac{\Delta t}{1+\Delta t/t_p}$",fontsize=14)
plt.ylabel(r"$p_{ws}-p_{wf}$",fontsize=14)

x2 = np.linspace(1,100)
y2 = 0.070*np.log10(x2)+3.453

plt.xlim((1,100))
plt.ylim((2000,5000))

# plt.plot(x2,10**y2,color='red')

plt.xscale("log")
plt.yscale("log")

plt.tight_layout()

plt.show()