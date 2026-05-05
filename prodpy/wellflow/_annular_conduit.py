import numpy as np

from ._circular_conduit import CircularConduit

class AnnularConduit(CircularConduit):

    def __init__(self, Do: float, Di: float, h: float):

        super().__init__(Do,h)

        self.__inner = CircularConduit(Di, self._height_correction(Do, Di, h))

        if self.__inner.D >= self.D:
            raise ValueError("Inner Diameter Di must satisfy 0 < Di < Do.")
        
    @staticmethod
    def _height_correction(Do: float, Di: float, h: float) -> float:
        """Correct the sand height for the inner pipe in an annular geometry."""
        return min(max(h - (Do - Di)/2, 0.), Di)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"  - The inner diameter of the outer conduit: {self.D}\n"
            f"  - The outer diameter of the inner conduit: {self.__inner.D}\n"
            f"  - The height of lower phase: {self.h}"
        )

    @property
    def P(self) -> float:
        """Returns float: Circle circumference (P_i + P_o)."""
        return np.pi * (self.D + self.__inner.D)
    
    @property
    def A(self) -> float:
        """Returns float : Annular cross section area."""
        return np.pi * (self.D**2 - self.__inner.D**2) / 4
    
    def _height2chord(self, height:float) -> float:
        """Returns float: The boundary between the segments excluding the inner circle's chord."""
        height_inner = self.height_correction(height)

        return super()._height2chord(height) - self.__inner._height2chord(height_inner)
    
    def _height2arc(self, height:float) -> float:
        """Convert segment height to the arc length of the lower segment."""
        height_inner = self.height_correction(height)

        return super()._height2arc(height) + self.__inner._height2arc(height_inner)
    
    def _height2area(self, height: float) -> float:
        """Returns the area of the lower segment based on its height."""
        height_inner = self.height_correction(height)

        return super()._height2area(height) - self.__inner._height2area(height_inner)
    
    def height_correction(self, height: float) -> float:
        """Returns the height for the inner pipe, referencing its lowest point."""
        return self._height_correction(self.D, self.__inner.D, height)
