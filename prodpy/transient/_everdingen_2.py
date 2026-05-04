import numpy as np

from scipy.sparse import csr_matrix as csr

from scipy.special import expi

from scipy.special import j0 as BJ0
from scipy.special import j1 as BJ1
from scipy.special import y0 as BY0
from scipy.special import y1 as BY1

from scipy.special import jvp as BJVp
from scipy.special import yvp as BYVp

from scipy.optimize import root_scalar

class Everdingen():
    """The solution based on the paper published by Everdingen et al."""

    def __init__(self,rr,tt,RR,num_of_terms=2):

        self.rr = np.array(rr).flatten()
        self.tt = np.array(tt).flatten()

        self.RR = RR

        self.num = num_of_terms

        self.find_roots()

    def find_roots(self):

        roots = np.empty(self.num)

        for idx in range(self.num):

            lower_bound = ((2*idx+1)*np.pi)/(2*self.RR-2)
            upper_bound = ((2*idx+3)*np.pi)/(2*self.RR-2)

            bracket = (lower_bound,upper_bound)

            solver = root_scalar(self.root_function,method="brentq",bracket=bracket)

            roots[idx] = solver.root

        self.beta_n = roots

    def root_function(self,beta):
        """
        This is the function that outputs values of root function 
        defined in Everdingen solution. At singularity point of \beta = 0,
        it outputs its value at limit.
        """

        beta = np.array(beta).flatten()

        res = np.empty(beta.shape)

        res[beta==0] = -self.RR/np.pi+1/(np.pi*self.RR)

        J1B = BJ1(beta[beta>0])
        Y1B = BY1(beta[beta>0])
        
        J1BR = BJ1(beta[beta>0]*self.RR)
        Y1BR = BY1(beta[beta>0]*self.RR)
        
        res[beta>0] = J1BR*Y1B-J1B*Y1BR

        return res

    def root_function_first_derivative(self,beta):
        """
        it needs a treshold value of beta and two functions to calculate analytical values,
        one close to singularity \beta=0, the other at larger values of \betta
        """

        J1B = BJ1(beta)
        Y1B = BY1(beta)
        
        J1BR = BJ1(beta*self.RR)
        Y1BR = BY1(beta*self.RR)

        J1B_prime = BJVp(1,beta)
        Y1B_prime = BYVp(1,beta)

        J1BR_prime = self.RR*BJVp(1,beta*self.RR)
        Y1BR_prime = self.RR*BYVp(1,beta*self.RR)

        return J1BR_prime*Y1B+J1BR*Y1B_prime-J1B_prime*Y1BR-J1B*Y1BR_prime

    def root_function_first_derivative_numerical(self,beta):
       
        num = beta.shape[0]

        idx = list(range(num))

        idx_diag_ends = np.array([0,num-1])

        Amatrix = csr((num,num))

        Amatrix += csr((np.ones(num-1),(idx[:-1],idx[1:])),shape=(num,num))
        Amatrix -= csr((np.ones(num-1),(idx[1:],idx[:-1])),shape=(num,num))
        Amatrix += csr((np.array([-1,1]),(idx_diag_ends,idx_diag_ends)),shape=(num,num))

        y_prime = Amatrix*root_function(beta,self.RR)
        x_prime = Amatrix*beta

        return y_prime/x_prime

    def solve(self):

        dist = self.rr.reshape((-1,1,1))
        time = self.tt.reshape((1,-1,1))
        beta = self.beta_n.reshape((1,1,-1))

        term1 = 2/(self.RR**2-1)*((dist[:,:,0]**2)/4.+time[:,:,0])
        term2 = self.RR**2/(self.RR**2-1)*np.log(dist[:,:,0])
        term3 = 3*self.RR**4-4*self.RR**4*np.log(self.RR)-2*self.RR**2-1
        term4 = 4*(self.RR**2-1)**2

        term5 = (BJ1(beta*self.RR))**2*np.exp(-(beta**2)*time)
        term6 = BJ1(beta)*BY0(beta*dist)-BY1(beta)*BJ0(beta*dist)
        term7 = beta*((BJ1(beta*self.RR))**2-(BJ1(beta))**2)

        term8 = term5*term6/term7

        self.PP = term1-term2-term3/term4+np.pi*term8.sum(axis=2)
        
if __name__ == "__main__":

    import unittest

    from tests import test_porous_media

    unittest.main(test_porous_media)
