import numpy as np

class RadialGrids():
    """Gridding class for the radial-cylindrical flow geometry."""
    
    def __init__(self,rw:float,re:float,num:int,tdelta:np.ndarray=None,zdelta:float|np.ndarray=None):

        self._num = num

        self.rw = rw
        self.re = re

        self.gamma = None

        self.r0 = None
        
        self.radius = None
        self.rdelta = None
        self.tdelta = tdelta
        self.zdelta = zdelta

    @property
    def num(self):
        return self._num

    @property
    def rw(self):
        return self._rw/0.3048
    
    @rw.setter
    def rw(self,value):
        self._rw = value*0.3048
    
    @property
    def re(self):
        return self._re/0.3048
    
    @re.setter
    def re(self,value):
        self._re = value*0.3048

    @property
    def gamma(self):
        return self._gamma
    
    @gamma.setter
    def gamma(self,value):
        self._gamma = (self._re/self._rw)**(1/self.num)

    @property
    def r0(self):
        return self._r0/0.3048
    
    @r0.setter
    def r0(self,value):
        self._r0 = (self._rw*np.log(self.gamma)/(1-1/self.gamma)).tolist()

    @property
    def radius(self):
        return self._radius/0.3048
    
    @radius.setter
    def radius(self,value):
        self._radius = self._r0*self.gamma**np.arange(-1,self.num+1)

    @property
    def rdelta(self):
        return self._rdelta/0.3048

    @rdelta.setter
    def rdelta(self,value):
        self._rdelta = self._radius[1:]-self._radius[:-1]

    @property
    def tdelta(self):
        """Returns theta-direction grid cell radian."""
        return self._tdelta

    @tdelta.setter
    def tdelta(self,value):
        """Sets theta-direction grid cell radian."""
        self._tdelta = 2*np.pi if value is None else np.ravel(value).astype(float)

    @property
    def zdelta(self):
        """Returns z-direction grid cell size in feet."""
        return self._zdelta/0.3048

    @zdelta.setter
    def zdelta(self,value):
        """Sets z-direction grid cell size after converting from feet to meters."""
        self._zdelta = 1 if value is None else np.ravel(value).astype(float)*0.3048

    # def prev_init(lengths):

    #     self.lengths = lengths

    #     numverts = 50

    #     thetas = np.linspace(0,2*np.pi,numverts+1)[:-1]

    #     self.edge_vertices = np.zeros((2*numverts,3))

    #     self.edge_vertices[:,0] = np.tile(self.lengths[0]/2*np.cos(thetas),2)+self.lengths[0]/2
    #     self.edge_vertices[:,1] = np.tile(self.lengths[1]/2*np.sin(thetas),2)+self.lengths[1]/2
    #     self.edge_vertices[:,2] = np.append(np.zeros(numverts),self.lengths[2]*np.ones(numverts))

    #     indices = np.empty((2*numverts,2),dtype=int)

    #     vertices_0 = np.arange(numverts)
    #     vertices_1 = np.append(np.arange(numverts)[1:],0)

    #     indices[:,0] = np.append(vertices_0,vertices_0+numverts)
    #     indices[:,1] = np.append(vertices_1,vertices_1+numverts)

    #     x_aspects = self.edge_vertices[:,0][indices]
    #     y_aspects = self.edge_vertices[:,1][indices]
    #     z_aspects = self.edge_vertices[:,2][indices]

    #     self.boundaries = []

    #     for x_aspect,y_aspect,z_aspect in zip(x_aspects,y_aspects,z_aspects):
    #         self.boundaries.append(np.array([x_aspect,y_aspect,z_aspect]))

    def grids(self):
        pass

    def table(self):
        pass

def plot_grid(r0,r1,N):

    gamma = (r1/r0)**(1/N)
    
    radius = r0*gamma**np.arange(N+1)

    return radius

def grid_concentric_old(r0,r1,r2,N):

    ratio = np.log(r2/r1)/np.log(r1/r0)

    N1 = N / (ratio+1)
    N2 = N - N1

    if N1 < N2:
        N1 = int(np.ceil(N1))
        N2 = N - N1
    else:
        N2 = int(np.ceil(N2))
        N1 = N - N2

    gamma1 = (r1/r0)**(1/N1)
    gamma2 = (r2/r1)**(1/N2)

    radius1 = r0*gamma1**np.arange(N1+1)
    radius2 = r1*gamma2**np.arange(N2+1)

    return radius1, radius2

def grid_concentric(radii, N):
    """
    Generate a radial grid for concentric cylindrical reservoir regions.

    Parameters
    ----------
    radii : list or array-like
        Region boundary radii, e.g. [rw, rd, re].
        Must be strictly increasing.
    N : int
        Total number of radial grid cells.

    Returns
    -------
    radius : np.ndarray
        One array of radial grid boundaries.
        Length will be N + 1.
    """

    radii = np.asarray(radii, dtype=float)

    if len(radii) < 2:
        raise ValueError("At least two radii are required.")

    if np.any(radii <= 0):
        raise ValueError("All radii must be positive.")

    if np.any(np.diff(radii) <= 0):
        raise ValueError("Radii must be strictly increasing.")

    n_regions = len(radii) - 1

    if N < n_regions:
        raise ValueError(
            f"N must be at least {n_regions}, so each region gets at least one cell."
        )

    # Allocate number of cells proportional to logarithmic radial thickness
    log_lengths = np.log(radii[1:] / radii[:-1])
    raw_cells = N * log_lengths / log_lengths.sum()

    # Start with at least one cell per region
    cells = np.floor(raw_cells).astype(int)
    cells = np.maximum(cells, 1)

    # Adjust total number of cells to exactly N
    while cells.sum() < N:
        fractions = raw_cells - np.floor(raw_cells)
        idx = np.argmax(fractions)
        cells[idx] += 1

    while cells.sum() > N:
        idx_candidates = np.where(cells > 1)[0]
        idx = idx_candidates[np.argmax(cells[idx_candidates])]
        cells[idx] -= 1

    # Build grid region by region
    grid = []

    for i in range(n_regions):
        r_in = radii[i]
        r_out = radii[i + 1]
        n = cells[i]

        gamma = (r_out / r_in) ** (1 / n)

        region_grid = r_in * gamma ** np.arange(n + 1)

        # Avoid duplicating internal boundaries
        if i > 0:
            region_grid = region_grid[1:]

        grid.extend(region_grid)

    return np.asarray(grid)

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # rad = RadialGrids(0.25,1000,4)

    # print(rad.gamma)
    # print(rad.r0)
    # print(rad.radius)
    # print(rad.rdelta)

    continuous = plot_grid(r0=0.1,r1=10,N=5)
    sub1,sub2 = grid_concentric_old(r0=0.1,r1=8,r2=10,N=5)
    main = grid_concentric(radii=[0.1,8,10],N=10)

    print(continuous.shape)
    print(main.shape)

    plt.plot(continuous,np.zeros_like(continuous),'o-')
    plt.plot(sub1,np.zeros_like(sub1)+5,'s-')
    plt.plot(sub2,np.zeros_like(sub2)+5,'^-')
    plt.plot(main,np.zeros_like(main)-5,'o-')

    plt.xscale('log')

    plt.xlabel('Radius (m)')
    plt.title('Radial Grid')
    plt.grid()
    plt.show()