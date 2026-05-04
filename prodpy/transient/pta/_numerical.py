import numpy as np

from scipy.sparse import csr_matrix as csr
from scipy.sparse.linalg import spsolve as sps

class finite():

    def __init__(self,R,rskin=1,alpha=1,Nimpaired=0,Nunharmed=50):
        """
        R         : reservoir radius

        rskin     : radius of impaired zone

        alpha     : permeability reduction in the impaired zone,
                    k_impaired/k_unharmed

        Nimpaired : number of grids in the impaired zone
        Nunharmed : number of grids in the unharmed zone
        """

        self.R = R

        self.rskin = rskin
        self.alpha = alpha

        self.Nimpaired = int(Nimpaired)
        self.Nunharmed = int(Nunharmed)

        self.radii_grid = self.setradii()
        self.radii_node = self.setnodes()

        self.alpha_grid = self.gridalpha()
        self.alpha_node = self.nodealpha()

        self.betaL,self.betaR = self.betas()

        self.delta1,self.delta2 = self.deltas()

    def pressure(self,tD,CD=0):
        """
        tD        : times to calculate well pressure
        CD        : dimensionless storage
        """

        #creating time steps
        timesteps = tD[1:]-tD[:-1]
        timesteps = np.insert(timesteps,0,timesteps[0])

        #initialization of pressure arrays
        press = np.zeros(self.size)
        pwell = np.zeros(timesteps.size+1)

        N,indices = self.size,self.indices

        for index,timestep in enumerate(timesteps,start=1):

            betaC = self.betaL+self.betaR+1/timestep

            theta1,theta2 = self.thetas(CD,timestep,pwell[index-1])

            betaC[0] -= theta2*self.betaL[0] # inner boundary correction
            betaC[-1] -= self.betaR[-1] # outer boundary correction

            Tmat = csr((N,N))
            
            Tmat += csr((betaC,(indices,indices)),shape=(N,N))
            Tmat += csr((-self.betaL[1:],(indices[1:],indices[:-1])),shape=(N,N))
            Tmat += csr((-self.betaR[:-1],(indices[:-1],indices[1:])),shape=(N,N))

            boundary = press/timestep

            boundary[0] += self.betaL[0]*theta1

            press = sps(Tmat,boundary)

            pimag = theta1+theta2*press[0]

            pwell[index] = self.delta1*press[0]+self.delta2*pimag

        return pwell[1:]

    def setradii(self):
        
        R,r = self.R,self.rskin

        Nimpaired = self.Nimpaired
        Nunharmed = self.Nunharmed

        Nimpaired_temp = int(np.ceil(r-1).tolist())

        if Nimpaired_temp == 0:
            Nimpaired = 0

        if Nimpaired == 0:
            return self.loggrid(1,R,Nunharmed)

        impaired,gamma1 = self.loggrid(1,r,Nimpaired,ratioFlag=True)

        node = self.logmean((impaired[-1],impaired[-1]*gamma1))

        unharmed = self.loggrid(node,R,Nunharmed-1)

        return np.append(impaired,unharmed[1:])

    def setnodes(self):

        return self.logmean(self.radii_grid)

    def gridalpha(self):

        array = np.ones(self.radii_grid.size)

        array[self.radii_grid<self.rskin] = self.alpha

        return array

    def nodealpha(self):

        radrad = np.log(self.radii_grid[1:]/self.radii_grid[:-1])
        nodrad = np.log(self.radii_node/self.radii_grid[:-1])/self.alpha_grid[:-1]
        radnod = np.log(self.radii_grid[1:]/self.radii_node)/self.alpha_grid[1:]

        return radrad/(nodrad+radnod)

    def betas(self):

        nwidth = self.width(self.radii_grid)
        gwidth = self.width(self.radii_node)

        nbeta = self.alpha_node*self.radii_node/nwidth

        betaL = nbeta[:-1]/self.radii_grid[1:-1]/gwidth
        betaR = nbeta[1:]/self.radii_grid[1:-1]/gwidth

        return betaL,betaR

    def deltas(self):

        nwidth = self.radii_grid[1]-self.radii_grid[0]

        delta1 = (1-self.radii_grid[0])/nwidth
        delta2 = (self.radii_grid[1]-1)/nwidth

        return delta1,delta2

    def thetas(self,storage,timestep,pwell):

        term1 = storage/timestep

        gwidth = self.radii_node[1]-self.radii_node[0]

        term2 = self.betaL[0]*self.radii_grid[1]*gwidth

        theta1 = (1+term1*pwell)/(term2+term1*self.delta2)
        theta2 = (term2-term1*self.delta1)/(term2+term1*self.delta2)

        return theta1,theta2

    @property
    def skin(self):

        return (1/self.alpha-1)*np.log(self.rskin)

    @property
    def size(self):

        return self.radii_grid.size-2

    @property
    def indices(self):

        return np.arange(self.size)

    @staticmethod
    def loggrid(start,stop,number=50,ratioFlag=False):

        centers = np.zeros(number+2)

        ratio = (stop/start)**(1/number)

        centers[1] = start*np.log(ratio)/(1-1/ratio)
        centers[0] = centers[1]/ratio

        for i in range(1,number+1):
            centers[i+1] = centers[i]*ratio

        if ratioFlag:
            return centers,ratio

        return centers

    @staticmethod
    def logmean(radii):

        radii = np.array(radii)

        return (radii[1:]-radii[:-1])/np.log(radii[1:]/radii[:-1])

    @staticmethod
    def width(radii):

        radii = np.array(radii)

        return radii[1:]-radii[:-1]

if __name__ == "__main__":

    from matplotlib import pyplot as plt

    print(finite.setnodes(10,r=2.5))

    # tD = 1e-2

    # umin = np.sqrt(1/tD)/1e6
    # umax = np.sqrt(1/tD)*1e5

    # u = np.logspace(np.log10(umin),np.log10(umax),2000)

    # y1 = everdingen.pressure_integrand(u,tD)
    # y2 = everdingen.pressure_integrand(u,1e8)
    # y1 = everdingen.pressure_integrand(u,0.0001)

    # y1 = agarwal.pressure_integrand(u,1e8,1e5,0)
    # y2 = agarwal.pressure_integrand(u,1e8,1e5,0)
    # y2 = agarwal.pressure_lineSource_integrand(u,1e3,10000,0)
    # y3 = agarwal.pressure_lineSource_integrand(u,1e3,10000,5)
    # y4 = agarwal.pressure_lineSource_integrand(u,1e3,10000,10)
    # y5 = agarwal.pressure_lineSource_integrand(u,1e3,10000,20)

    # print(f"{agarwal.pressure(1e8,1e5,0)[0]:8.5f}")

    # plt.loglog(u,y1)
    # plt.loglog(u,y1)
    # plt.loglog(u,y2)
    # plt.loglog(u,y3)
    # plt.loglog(u,y4)
    # plt.loglog(u,y5)

    # plt.show()



