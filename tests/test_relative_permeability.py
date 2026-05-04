import unittest

import numpy as np

if __name__ == "__main__":
    import dirsetup

from flow.pormed import RelPerm

class TestRelativePermeability(unittest.TestCase):
    
    def test_oil_water_imbibition(self):

        """
        Reservoir Engineering Handbook Second Edition Tarek Ahmed
        Chapter 5 - Relative Permeability Concepts
        Two Phase Relative Permeability from Analytical Equations
        Example - page 297 - Oil-Water System
        """

        Sw = np.array([0.25,0.3,0.4,0.5,0.6,0.65])
        
        RP = RelPerm(Sorow=0.35,Swc=0.25,krowc=0.85,krwor=0.4,no=0.9,nw=1.5)

        RP.krw,RP.kro = RP.water_oil(Sw)

        kro = np.array([0.850,0.754,0.557,0.352,0.131,0.000])
        krw = np.array([0.000,0.018,0.092,0.198,0.327,0.400])

        np.testing.assert_array_almost_equal(kro,RP.kro,decimal=3)
        np.testing.assert_array_almost_equal(krw,RP.krw,decimal=3)

    def test_gas_oil_imbibition(self):

        """
        Reservoir Engineering Handbook Second Edition Tarek Ahmed
        Chapter 5 - Relative Permeability Concepts
        Two Phase Relative Permeability from Analytical Equations
        Example - page 297 - Gas-Oil System
        """

        Sg = np.array([0.05,0.1,0.2,0.3,0.4,0.52])

        RP = RelPerm(Sorgo=0.23,Swc=0.25,Sgc=0.05,krogc=0.6,krglc=0.95,no=1.2,ng=0.6)

        RP.krg,RP.kro = RP.gas_oil(Sg)

        kro = np.array([0.600,0.524,0.378,0.241,0.117,0.000])
        krg = np.array([0.000,0.248,0.479,0.650,0.796,0.950])

        np.testing.assert_array_almost_equal(kro,RP.kro,decimal=3)
        np.testing.assert_array_almost_equal(krg,RP.krg,decimal=3)

    def three_phase_system(self):

        pass
                       
if __name__ == "__main__":

    unittest.main()
