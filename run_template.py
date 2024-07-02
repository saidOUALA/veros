import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from veros.setups.acc import ACCSetup
#from veros.setups.acc_learning import ACCLearningSetup
#from veros.setups.acc_variable_res import ACCResSetup

simulation = ACCSetup()
simulation.setup()
simulation.run()

# compare here the energy/avg/ACC transport time series to the ones provided in the repo

# ACC diag:
acc_diag = nc.Dataset(r"acc_restart_from_50000_test_diag.acc_diags.nc")
plt.plot(acc_diag.variables['flux_north_south'][:])
plt.show()
plt.imshow(acc_diag.variables['temp_zonal_mean'][:][-1])
plt.colorbar()
plt.show()
plt.imshow(acc_diag.variables['salt_zonal_mean'][:][-1,:,:])
plt.colorbar()
plt.show()


plt.imshow(simulation.state.variables.temp[:,:,-1,1], cmap = 'jet')
plt.colorbar()
plt.show()


plt.imshow(simulation.state.variables.v[:,:,-1,1], cmap = 'jet')
plt.colorbar()
plt.show()

plt.imshow(simulation.state.variables.temp[3:7,:,:,1].mean(axis = 0), cmap = 'jet')
plt.colorbar()
plt.show()

plt.imshow(simulation.state.variables.temp[3,:,:,1]+simulation.state.variables.temp[5,:,:,1], cmap = 'jet')
plt.colorbar()
plt.show()

plt.imshow(simulation.state.variables.surface_taux, cmap = 'jet')
plt.colorbar()
plt.show()

plt.imshow(simulation.state.variables.temp[:,:,-1,0], cmap = 'jet')
plt.colorbar()
plt.show()



plt.imshow(simulation.state.variables.salt[:,:,:,0].mean(axis = 0))
plt.colorbar()
plt.show()


# averages:
acc_avg = nc.Dataset(r"acc_restart_from_60000.averages.nc")
plt.imshow(acc_avg.variables['u'][-1,-1,:, :].T, cmap = 'jet')
plt.colorbar()
plt.show()


acc_nrj = nc.Dataset(r"acc_restart_from_30000.energy.nc")
plt.plot(acc_nrj.variables['Time'][:],acc_nrj.variables['k_m'][:])
plt.show()


acc_ovr = nc.Dataset(r"acc_restart_from_30000.overturning.nc")



"""
plt.imshow(simulation.state.variables.u[:,:,0, 0], cmap = 'jet')
plt.colorbar()
plt.show()

vm_T = nc.Dataset(r"acc_t.training.nc")


vm_T.variables['u'][:].data[(vm_T.variables['u'][:].mask)]
simulation.state.variables.u[:,:,:, 1].T[(vm_T.variables['u'][-1,:,:,:].mask)].max()

plt.imshow(simulation.state.variables.maskU[:,:, 0], cmap = 'jet')
plt.colorbar()
plt.show()

u_nc = vm_T.variables['u'][:].filled(fill_value=np.nan)

plt.imshow(simulation.state.variables.u[:,:,0, 0]-u_nc[-2,0,:,:].T, cmap = 'jet')
plt.colorbar()
plt.show()
"""