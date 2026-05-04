import numpy

from scipy.sparse import csr_matrix as csr

from ._vector import Vector
from ._matrix import Matrix

class Builder():

    def __init__(self,grids):
        """Initialization of matrix building class."""
        self.grids = grids

    def __getattr__(self,key):
        """Delegates attribute access to the underlying grid object."""
        return getattr(self.grids,key)

    @property
    def square(self):
        """Returns the shape of square matrices in the flow calculations."""
        return (self.nums,)*2

    @property
    def column(self):
        """Returns the shape of column matrices in the flow calculations."""
        return (self.nums,1)

    @property
    def filler(self):
        return Filler
    
    def matrix(self,vec:Vector):
        """Returns matrices to be used in the solver."""
        A = self.Amat(vec)
        T = self.Tmat(vec)
        G = self.Gmat(T)
        J = self.Jmat(vec)
        Q = self.Qmat(vec)

        return Matrix(A,T,G,J,Q)

    def Amat(self,vec:Vector):
        """Returns Accumulation matrix filled with diagonal values."""
        return csr((vec._A,(self.index,self.index)),shape=self.square)

    def Tmat(self,vec:Vector):
        """Returns Transmissibility matrix filled with diagonal and offset values."""
        matrix = csr(self.square)

        matrix = self.filler.tmatrix(matrix,vec._X,self.xneg,self.xpos)
        matrix = self.filler.tmatrix(matrix,vec._Y,self.yneg,self.ypos)
        matrix = self.filler.tmatrix(matrix,vec._Z,self.zneg,self.zpos)

        return matrix

    def Gmat(self,tmatrix):
        """Returns Gravity column matrix filled with gravity related terms."""
        return tmatrix.dot(self._hhead.reshape((-1,1)))

    def Jmat(self,vec:Vector):
        """Returns J matrix filled with constant pressure constraints on diagonal."""
        matrix = csr(self.square)

        for well in vec._W:
            matrix += self.filler.jmatrix(matrix,well)

        for edge in vec._B:
            matrix += self.filler.jmatrix(matrix,edge)

        return matrix

    def Qmat(self,vec:Vector):
        """Returns Q column matrix filled with constraints."""
        matrix = csr(self.column)

        for well in vec._W:
            matrix += self.filler.qmatrix(matrix,well)

        for edge in vec._B:
            matrix += self.filler.qmatrix(matrix,edge)

        return matrix

class Filler:

    @staticmethod
    def tmatrix(matrix:csr,values,neg,pos):
        """Retuns updated T square matrix."""
        matrix += csr((values,(neg,neg)),shape=matrix.shape)
        matrix -= csr((values,(neg,pos)),shape=matrix.shape)

        matrix += csr((values,(pos,pos)),shape=matrix.shape)
        matrix -= csr((values,(pos,neg)),shape=matrix.shape)

        return matrix

    @staticmethod
    def jmatrix(matrix:csr,constr):
        """Returns updated J diagonal matrix."""
        if constr.sort=="press":
            vector = (constr._prod,(constr.index,constr.index))
            matrix = csr(vector,shape=matrix.shape)

        return matrix

    @staticmethod
    def qmatrix(matrix:csr,constr):
        """Returns updated Q column matrix."""
        vector = constr._prod*constr._cond

        if constr.sort!="press":
            vector /= constr._prod.sum()

        vector = (vector,(constr.index,numpy.zeros_like(constr.index)))
        matrix = csr(vector,shape=matrix.shape)

        return matrix
        
if __name__ == "__main__":

    pass