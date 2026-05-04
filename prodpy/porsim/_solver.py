from scipy.sparse import csr_matrix as csr
from scipy.sparse import linalg

from ._block import Block

class BaseSolver(Block):
    """
    The Base Class initializing reservoir flow in Rectangular Cuboids;
    """

    def __init__(self,grids,rrock,fluid,tcomp=None):
        """
        Initialization of Base Class Solver.

        Properties:
        ----------
        grids  : It is a GridDelta instance.

        rrock  : It is a ResRock instance providing rock propertie
            at any given pressure (or time step).

        fluid  : It is a Fluid instance providing fluid properties
            at any given pressure (or time step).

        tcomp  : total compressibility, 1/psi

        """
        super().__init__(grids,rrock,fluid,tcomp)

    @property
    def iterator(self):
        return Iterator

    @property
    def residual(self):
        return Residual

class Iterator:

    @staticmethod
    def explicit(mat,Pn):
        """Explicit pressure solution returning P_{n+1}."""
        RHS = csr.dot(-(mat._T+mat._J),Pn)+mat._Q+mat._G

        return Pn+linalg.spsolve(mat._A,RHS)

    @staticmethod
    def mixed(mat,Pn,theta:float=0.5):
        """Mixed pressure solution returning P_{n+1}."""
        LHS = (1-theta)*(mat._T+mat._J)+mat._A
        RHS = csr.dot(mat._A-theta*(mat._T+mat._J),Pn)+mat._Q+mat._G

        return linalg.spsolve(LHS,RHS)

    @staticmethod
    def implicit(mat,Pn):
        """Implicit pressure solution returning P_{n+1}."""
        RHS = csr.dot(mat._A,Pn)+mat._Q+mat._G
        
        return linalg.spsolve(mat._T+mat._J+mat._A,RHS)

class Residual:

    @staticmethod
    def explicit(mat,Pn,P):
        """Returns residual vector in SI units, (m**3)/(sec)"""
        return -csr.dot(mat._A,P)+csr.dot(mat._A-(mat._T+mat._J),Pn)+mat._Q+mat._G

    @staticmethod
    def implicit(mat,Pn,P):
        """Returns residual vector in SI units, (m**3)/(sec)"""
        return -csr.dot(mat._T+mat._J+mat._A,P)+csr.dot(mat._A,Pn)+mat._Q+mat._G