import numpy as np

from typx.wtest import reservoir,well,fluid
from typx.wtest import pressure

res = reservoir(permeability=200,porosity=0.15,thickness=30)

oil = fluid(viscosity=1.5,formation_volume_factor=1.2)

prod = well(rate=800,time=16,radius=0.25,skin=0)

P = pressure(res,oil,prod)

P.set_compressibility(total=25e-6)

P.initialize(scale='log')

P.set_boundary(area=200)

P.view(scale="linear")
