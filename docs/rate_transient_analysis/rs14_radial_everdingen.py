import matplotlib.pyplot as plt
import numpy as np

from scipy.special import expi

from scipy.sparse import csr_matrix as csr

from scipy.special import j0 as bessel_j0
from scipy.special import j1 as bessel_j1
from scipy.special import y0 as bessel_y0
from scipy.special import y1 as bessel_y1

from scipy.special import jvp as bessel_jvp
from scipy.special import yvp as bessel_yvp

from scipy.optimize import root_scalar

from borepy.pormed import OnePhaseRadial

beta = np.logspace(-4,3,10000)
u = 1e-2
# print(bessel_y0(u))
# print(2/np.pi*(np.log(u)-0.11593))
beta = np.linspace(0,100,100000)

tt = np.linspace(8,40)
tt = [0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2,0.22,0.24,0.26,0.28,0.3,0.35,0.4,0.45,0.5,0.55,0.6]
tt = np.linspace(0.22,0.5,15)
tt = np.linspace(0.4,0.6,11)
tt = np.linspace(1,2,11)
tt = np.linspace(1e-2,50,9)

tt = [1e-2,1e-1,0.5,3,5,7,9,11,16]

##    print(tt)

sol = OnePhaseRadial(rr=np.linspace(1,10,10000),tt=tt,RR=10,num_of_terms=100) #np.linspace(1,10)

sol.solve()

##    print(sol.beta_n)
##    print(sol.PP[0,:])

plt.plot(sol.rr,sol.PP[:,0])
plt.plot(sol.rr,sol.PP[:,1])
plt.plot(sol.rr,sol.PP[:,2])
plt.plot(sol.rr,sol.PP[:,3])
plt.plot(sol.rr,sol.PP[:,4])
plt.plot(sol.rr,sol.PP[:,5])
plt.plot(sol.rr,sol.PP[:,6])
plt.plot(sol.rr,sol.PP[:,7])
plt.plot(sol.rr,sol.PP[:,8])

plt.gca().invert_yaxis()

plt.show()

##    J0B = bessel_j0(sol.beta_n)
##    Y0B = bessel_y0(sol.beta_n)
##
##    J1B = bessel_j1(sol.beta_n)
##    Y1B = bessel_y1(sol.beta_n)

##    print(np.pi*sol.beta_n*(J1B*Y0B-Y1B*J0B))

##    print(sol.beta_n)

# P = OnePhaseRadial(10,1000)

# f0 = P.root_function(R)
# f1A = P.root_function_first_derivative(R)
# f1N = P.root_function_first_derivative_numerical(R)

# f0 = root_function(1e-20)

# x, fsol, XE, FE, IE = odelay(root_function_first_derivative,f0,beta, events=[e1])

# print(C[0]+D[0])

# plt.plot(beta,root_function(beta,R),label="root function")
# plt.plot(beta,root_function_first_derivative_numerical(beta,R),label="root function derivative")

##    plt.plot(beta,sol.root_function(beta),label="J1BR*Y1B-J1B*Y1BR")
##    plt.plot(sol.beta_n,np.zeros(sol.beta_n.shape),'.')
# plt.semilogy(beta,np.abs(B),label="J1BR*Y1B_prime")
# plt.semilogy(beta,np.abs(C),label="J1B_prime*Y1BR")
# plt.semilogy(beta,np.abs(D),label="J1B*Y1BR_prime")

# plt.axhline(y=0,xmin=beta[0],xmax=beta[-1],color='k')
# plt.axhline(y=0,xmin=np.log10(beta[0]),xmax=np.log10(beta[-1]),color='k')

# plt.ylim([-10,10])

# plt.legend()

##    plt.show()
