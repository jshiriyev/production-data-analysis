import numpy as np

class GridRadial():
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

if __name__ == "__main__":

    rad = GridRadial(0.25,1000,4)

    print(rad.gamma)
    print(rad.r0)
    print(rad.radius)
    print(rad.rdelta)