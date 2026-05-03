import numpy

import matplotlib.pyplot as plt

from typx.wtest import everdingen1949

tD = [0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.5,2,2.5,3,4,5,6,7,8,9,10,15,20,25,30,40,50,60,70,80,90,100,150,200,250,300,400,500,600,700,800,900,1000]

pwD = everdingen1949.pressure(tD)

for time,p in zip(tD,pwD):
    print(f"{time:11,.3f} {p:8.4f}")
