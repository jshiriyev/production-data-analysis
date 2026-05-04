class Matrix():

    def __init__(self,A,T,G,J,Q):
        """Inputs should be in SI units.

        A   : Accumulation-Compressibility Matrix,
              (m**3)/(Pa*sec)

        T   : Inter-Block Transmissibility Matrix,
              (m**3)/(Pa*sec)

        G   : Gravity Related Column Matrix,
              (m**3)/(sec)

        J   : Constant Pressure Constraint Matrix,
              (m**3)/(Pa*sec)

        Q   : Flow Rate Constraint Column Matrix,
              (m**3)/(sec)
        
        """

        self._A = A
        self._T = T
        self._G = G
        self._J = J
        self._Q = Q
    
    @property
    def A(self):
        return self._A*(3.28084**3)*(24*60*60)*6894.76

    @property
    def T(self):
        """Converting from SI Units to Oil Field Units."""
        return self._T*(3.28084**3)*(24*60*60)*6894.76
    
    @property
    def G(self):
        """Converting from SI Units to Oil Field Units."""
        return self._G*(3.28084**3)*(24*60*60)

    @property
    def J(self):
        """Converting from SI Units to Oil Field Units."""
        return self._J*(3.28084**3)*(24*60*60)*6894.76
    
    @property
    def Q(self):
        """Converting from SI Units to Oil Field Units."""
        return self._Q*(3.28084**3)*(24*60*60)
    

if __name__ == "__main__":

    pass