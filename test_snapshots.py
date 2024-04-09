import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

vm = nc.Dataset(r"global_1deg.snapshot.nc")
vm.variables.keys()
plt.imshow(vm.variables['u'][:].data[-1,-1,:,:])
plt.show()