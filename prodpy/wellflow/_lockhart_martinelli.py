from respy import Fluid

from .pressure_drop._pipe import Pipe
from .pressure_drop._darcy_weisbach import DarcyWeisbach

DarcyWeisbach.LOWER_REYNOLDS_LIMIT = 1000
DarcyWeisbach.UPPER_REYNOLDS_LIMIT = 2000

class LockhartMartinelli():

    def __init__(self,pipe:Pipe,gas:Fluid,liq:Fluid):

        self.pipe = pipe
        
        self.gas = gas
        self.liq = liq

    @property
    def gas(self):
        """Getter for the superficial gas model."""
        return self._gas

    @gas.setter
    def gas(self,value):
        """Setter for the superficial gas model."""
        self._gas = DarcyWeisbach(self.pipe,value)

    @property
    def liq(self):
        """Getter for the superficial liquid model."""
        return self._liq
    
    @liq.setter
    def liq(self,value):
        """Setter for the superficial liquid model."""
        self._liq = DarcyWeisbach(self.pipe,value)

    def get(self,grate,lrate,gdict:dict=None,ldict:dict=None):
        pass

    @staticmethod
    def regime(lamG:bool,turbG:bool,lamL:bool,turbL:bool):
        """Returns flow regime for both phases"""
        if lamL and lamG:
            return True,True
        elif lamL and turbG:
            return True,False
        elif turbL and lamG:
            return False,True
        elif turbL and turbG:
            return False,False

        raise "Transition zone observed"
    
    @staticmethod
    def dispersed_bubbly(x):
        return 4-12*np.log(x)+28*(np.log(x))**2

    @staticmethod
    def elongated_bubbly(x):
        return 4-15*np.log(x)+26*(np.log(x))**2

    @staticmethod
    def smooth_stratified(x):
        return 2+4.5*np.log(x)+3.6*(np.log(x))**2

    @staticmethod
    def stratified_wavy(x):
        return 3+1.65*np.log(x)+0.45*(np.log(x))**2

    @staticmethod
    def slug_flow(x):
        return 2.2+6.5*np.log(x)

    @staticmethod
    def annular_mist(x):
        return 4+2.5*np.log(x)+0.5*(np.log(x))**2