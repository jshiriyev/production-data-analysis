import numpy as np
import pytest

from prodpy.resprops.rperm.models import ModifiedCorey

def test_oil_water_imbibition():
    """
    Reservoir Engineering Handbook Second Edition Tarek Ahmed
    Chapter 5 - Relative Permeability Concepts
    Two Phase Relative Permeability from Analytical Equations
    Example - page 297 - Oil-Water System
    """

    Sw = np.array([0.25,0.3,0.4,0.5,0.6,0.65])
    
    RP = ModifiedCorey(
        Swr = 0.25,
        Snwr = 0.35,
        krw0 = 0.4,
        krnw0 = 0.85,
        nw = 1.5,
        nnw = 0.9,
    )
    # RP = RelPerm(Sorow=0.35,Swc=0.25,krowc=0.85,krwor=0.4,no=0.9,nw=1.5)
    model_krw, model_kro = RP.get(Sw)
    # RP.krw,RP.kro =RP.water_oil(Sw)

    krw = np.array([0.000,0.018,0.092,0.198,0.327,0.400])
    kro = np.array([0.850,0.754,0.557,0.352,0.131,0.000])

    np.testing.assert_array_almost_equal(krw,model_krw,decimal=3)
    np.testing.assert_array_almost_equal(kro,model_kro,decimal=3)

def test_gas_oil_imbibition():
    """
    Reservoir Engineering Handbook Second Edition Tarek Ahmed
    Chapter 5 - Relative Permeability Concepts
    Two Phase Relative Permeability from Analytical Equations
    Example - page 297 - Gas-Oil System
    """

    Sg = np.array([0.05,0.1,0.2,0.3,0.4,0.52])

    RP = ModifiedCorey(
        Swr = 0.48,
        Snwr = 0.05,
        krw0 = 0.6,
        krnw0 = 0.95,
        nw = 1.2,
        nnw = 0.6,
    )

    model_kro, model_krg = RP.get(1 - Sg)

    kro = np.array([0.600,0.524,0.378,0.241,0.117,0.000])
    krg = np.array([0.000,0.248,0.479,0.650,0.796,0.950])

    np.testing.assert_array_almost_equal(kro,model_kro,decimal=3)
    np.testing.assert_array_almost_equal(krg,model_krg,decimal=3)

# def test_three_phase_system(self):
#     pass