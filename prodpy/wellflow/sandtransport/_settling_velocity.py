import math
import warnings

class SettlingVelocity:
    
    _g = 9.81 # gravitational acceleration

    def __init__(self,
        particle_density: float,   # particle density [kg/m^3]
        fluid_density: float,   # fluid density [kg/m^3]
        particle_diameter: float,     # particle diameter [m]
        fluid_viscosity: float,     # dynamic fluid_viscosity [Pa.s]
    ) -> None:
        
        self.particle_density = particle_density
        self.fluid_density = fluid_density
        self.particle_diameter = particle_diameter
        self.fluid_viscosity = fluid_viscosity

        self._validate_inputs()

    def _validate_inputs(self):

        if self.particle_density <= 0:
            raise ValueError("The particle density must be greater than zero.")
        if self.fluid_density <= 0:
            raise ValueError("The fluid density must be greater than zero.")
        if self.particle_density < self.fluid_density:
            raise ValueError("Particle density must be greater than or equal to fluid density for settling.")
        if self.particle_diameter <= 0:
            raise ValueError("The particle diameter must be greater than zero.")
        if self.fluid_viscosity <= 0:
            raise ValueError("Dynamic viscosity must be greater than zero.")

    @property
    def g(self):
        return self._g

    @property
    def rho_diff(self) -> float:
        return self.particle_density-self.fluid_density
    
    def reynolds_number(self, relative_velocity: float) -> float:
        if relative_velocity < 0:
            raise ValueError("The relative velocity between fluid and particle must be greater than or equal to zero.")
        
        return (self.fluid_density * relative_velocity * self.particle_diameter) / self.fluid_viscosity
    
    def drag_coefficient(self, particle_reynolds_number: float) -> float:

        if particle_reynolds_number <= 0:
            raise ValueError("Reynolds number must be greater than zero.")
        
        if particle_reynolds_number < 1:
            return (24 / particle_reynolds_number )

        if particle_reynolds_number < 1000:
            return (24 / particle_reynolds_number) * (1 + 0.15 * particle_reynolds_number**0.687)
        
        return 0.44
    
    def stokes_equation(self):
        return ( self.rho_diff * self.g * self.particle_diameter**2 ) / (18 * self.fluid_viscosity)
    
    def general_equation(self, drag_coefficient:float):
        if drag_coefficient <= 0:
            raise ValueError("Drag coefficient must be greater than zero.")
        
        return math.sqrt(
            (4 * self.particle_diameter * self.rho_diff * self.g) / (3 * drag_coefficient * self.fluid_density)
        )

    def __call__(self, tol: float = 1e-6, max_iter: int = 100) -> float:

        # --- Initial guess (Stokes as starting point) ---
        vs = self.stokes_equation()

        for _ in range(max_iter):

            # Reynolds number
            Rep = self.reynolds_number(vs)

            # Drag coefficient (Schiller-Naumann)
            C_d = self.drag_coefficient(Rep)

            # Updated settling velocity
            v_new = self.general_equation(C_d) 

            # Check convergence
            if abs(v_new - vs) < tol:
                return v_new

            vs = v_new

        else:
            warnings.warn(
                "Settling velocity did not converge within the maximum iterations.",
                RuntimeWarning
            )

        return vs # If not converged