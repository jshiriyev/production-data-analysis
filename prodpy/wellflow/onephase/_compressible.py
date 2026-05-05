from ._darcy_weisbach import DarcyWeisbach

class Compressible(DarcyWeisbach):

    def __init__(self,*args,**kwargs):

        super().__init__(*args,**kwargs)

    def get(self,P2,P1):

        rho1 = (P1*self.Fluid.molarweight[0])/(self.UGC*self.temperature)
        rho2 = (P2*self.Fluid.molarweight[0])/(self.UGC*self.temperature)

        nu1 = 1/rho1
        nu2 = 1/rho2

        Dp = P1**2-P2**2
        Rp = np.log(P1/P2)
        Ft = 4*self.phi*self.Pipe.length/self.Pipe.diameter
        
        G = self.Pipe.csa*np.sqrt(Dp/(2*P1*nu1*(Rp+Ft)))

        return G

    def omega(self):

        def objective(wc,LHS):
            RHS = (1/wc)**2-(np.log(1/wc))**2-1
            return (RHS-LHS)**2

        LHS = 8*self.phi*self.Pipe.length/self.Pipe.diameter

        self.omega = minimize_scalar(objective,args=(LHS,),bounds=((1e-5,1)),method='bounded').x