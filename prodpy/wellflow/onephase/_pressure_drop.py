from prodpy.resprops import Fluid
from prodpy.wellflow import Pipe

class PressureDrop():
    """A base class for pressure drop calculations in a pipe due to fluid flow."""

    def __init__(self,pipe:Pipe,fluid:Fluid):
        """
        Initializes the PressureDrop class.

        Args:
            pipe (Pipe): A Pipe object representing the pipe's geometry.
            fluid (Fluid): A Fluid object representing the fluid properties.

        """
        self.pipe, self.fluid = pipe, fluid