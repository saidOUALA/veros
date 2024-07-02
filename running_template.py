import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from veros.setups.acc import ACCSetup

simulation = ACCSetup()
simulation.setup()
simulation.run()

# compare here the energy/avg/ACC transport time series to the ones provided in the repo
