import numpy as np

class BaseClass():

    def __init__(self):
        pass

    @property
    def dr(self):
        """Returns an instance of the Drainage class."""
        pass

    @property
    def im(self):
        """Returns an instance of the Imbibition class."""
        pass

    @staticmethod
    def pc2jf(pc:float,perm:float,poro:float,ift:float,theta:float=0.):
        """Calculates J-function from capillary pressure values.
        
        Parameters:
        ----------
        pc     : capillary pressure values given for the following properties

        perm   : permeability
        poro   : porosity
        ift    : interfacial tension
        theta  : contact theta, radians

        """
        return pc*np.sqrt(perm/poro)/(ift*np.cos(theta))

    @staticmethod
    def jf2pc(jf,perm,poro,ift,theta=0.):
        """Calculates capillary pressure values from J-function.
        
        Parameters:
        ----------
        jf     : the Leverett J-function

        perm   : permeability
        poro   : porosity
        ift    : interfacial tension
        theta  : contact theta, radians

        """
        return jf*(ift*np.cos(theta))/np.sqrt(perm/poro)

def ScanningCurves(sw,hyst,cappres,epspc=0,residual=0):
    """Initialization of a reservoir is assumed to occur by primary drainage (oleic
    displaces aqueous phase) by migration from a source rock.

    During waterflooding, the capillary pressure transitions from a drainage curve to
    an imbibition curve.

    Capillary scanning curves are utilized in simulators to model this
    transition.

    A capillary scanning curve is a weighted average of the imbibition
    and drainage capillary pressure curves."""

    conds = sw<=hyst

    cappres = numpy.zeros(sw.shape)

    drainage = cappres.drainage(sw)

    imbibition = cappres.imbibition(sw)

    Av = 1-residual-hyst
    Bv = sw-hyst

    fv = (Av+epspc)/(Bv+epspc)*Bv/Av

    cappres[conds] = drainage

    cappres[~conds] = fv*imbibition+(1-fv)*drainage

    return cappres

if __name__ == "__main__":

    from _brooks_corey import BrooksCorey

    ow = BrooksCorey(0.1,0.4,2,3.5)

    pc1 = ow.drain.pc(0.3)
    pc2 = ow.imbibe.pc(0.3)

    jf = JFunction(pc1,100,0.2,20)
    print(pc1,jf.get(50,0.15,25))

    jf = JFunction(pc2,100,0.2,20)
    print(pc2,jf.get(50,0.15,25))