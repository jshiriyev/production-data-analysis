import logging

import numpy as np

from respy import Fluid

from .pressure_drop._pipe import Pipe
from .pressure_drop._darcy_weisbach import DarcyWeisbach

DarcyWeisbach.LOWER_REYNOLDS_LIMIT = 1000
DarcyWeisbach.UPPER_REYNOLDS_LIMIT = 2000

from ._lockhart_martinelli import LockhartMartinelli

class Chisholm(LockhartMartinelli):

    def __init__(self,pipe:Pipe,gas:Fluid,liq:Fluid):

        super().__init__(pipe,gas,liq)

    def get(self,grate,lrate,gdict:dict=None,ldict:dict=None):

        dropG = self.gas.get(grate,**(gdict or {}))
        dropL = self.liq.get(lrate,**(ldict or {}))

        X = self.get_X(dropG,dropL)
        C = self.get_C(
            self.gas.laminar,self.gas.turbulent,
            self.liq.laminar,self.liq.turbulent,
            )

        phiG = self.get_phiG(X,C)
        phiL = self.get_phiL(X,C)

        tdropG = phiG**2*dropG
        tdropL = phiL**2*dropL

        if np.abs(tdropG-tdropL)/np.abs(tdropL)>1e-5:
            logging.warning(f"Two-phase pressure drops calculated from liquid phase and gas phase are not the same: DP_G = {tdropG}, DP_L = {tdropL}")

        return tdropG

    @staticmethod
    def get_X(dropG,dropL):
        """Returns liquid to gas pressure drop ratio."""
        return np.sqrt(dropL/dropG)

    @staticmethod
    def get_C(lamG:bool,turbG:bool,lamL:bool,turbL:bool):
        """Returns C constant based on Chisholm method."""
        if lamL and lamG:
            return 5.
        elif turbL and lamG:
            return 10.
        elif lamL and turbG:
            return 12.
        elif turbL and turbG:
            return 20

        raise "Transition zone observed"

    @staticmethod
    def get_phiG(X,C):
        """Returns gas drag ratio."""
        return np.sqrt(1+C*X+X**2)

    @staticmethod
    def get_phiL(X,C):
        """Returns liquid drag ratio."""
        return np.sqrt(1+C/X+1/X**2)