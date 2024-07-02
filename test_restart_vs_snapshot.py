import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
#from veros.setups.global_1deg_learning import GlobalOneDegreeLearningSetup
from veros.setups.acc_learning import ACCLearningSetup
#from veros.setups.acc import ACCSetup

simulation = ACCLearningSetup()
simulation.setup()



# extract variables needed to restart simulation
restart_vars = {var: meta for var, meta in simulation.state.var_meta.items() if meta.write_to_restart and meta.active}

# construct a dictionary that contains those variables from the save simulation
vm_T = nc.Dataset(r"acc.training.nc")
training_simulation = {}
idx_training = 4
for key in restart_vars.keys():
    if key == 'taum1':
        training_simulation['taum1'] = 0#vm_T.variables[key][:][idx_training]#0
    elif key == 'tau':
        training_simulation['tau'] = 1#vm_T.variables[key][:][idx_training]#1
    elif key == 'taup1':
        training_simulation['taup1'] = 2#vm_T.variables[key][:][idx_training]#2
    elif key == 'time':
        training_simulation['time'] = 0
    elif key == 'K_diss_v':#key == 'Time' or key == 'time' or key == 'K_diss_v':
        training_simulation[key] = vm_T.variables[key][:].filled(fill_value=0.0)[idx_training].T
    else:
        training_simulation[key] = vm_T.variables[key][:].filled(fill_value=0.0)[idx_training-1:idx_training+2].T


from veros.io_tools import hdf5 as h5tools
from veros.restart import read_from_h5
with h5tools.threaded_io("acc_0010.restart.h5", "r") as infile, simulation.state.variables.unlock():
    # core restart
    restart_vars = {var: meta for var, meta in simulation.state.var_meta.items() if meta.write_to_restart and meta.active}
    _, restart_data = read_from_h5(simulation.state.dimensions, restart_vars, infile, "core", simulation.state.settings.enable_cyclic_x)



restart_data['u'].shape
restart_data['time']
vm_T.variables['time'][:]


np.abs(vm_T.variables['u'][:].filled(fill_value=0.0)[-1].T - restart_data['u'][:,:,:,1]).max()



simulation.set_initial_conditions_learning(simulation.state, training_simulation)
simulation.run()

var_data = variables.remove_ghosts(var_data, var.dims)[tmask].T


np.abs(simulation.state.variables.maskU*(simulation.state.variables.u[:,:,:,simulation.state.variables.tau]) - simulation.state.variables.u[:,:,:,simulation.state.variables.tau]).max()

np.abs(simulation.state.variables.v[:,:,:,simulation.state.variables.tau] - vm_T.variables['v'][:].filled(fill_value=0.0)[idx_training+2].T[:,:,:]).mean()
np.abs(simulation.state.variables.temp[:,:,:,simulation.state.variables.tau] - vm_T.variables['temp'][:].filled(fill_value=0.0)[idx_training+2].T[:,:,:]).max()

np.abs(simulation.state.variables.maskV[2:-2,2:-2,:]*(simulation.state.variables.v[2:-2,2:-2,:,simulation.state.variables.tau] - vm_T.variables['v'][:].filled(fill_value=0.0)[idx_training+2].T[2:-2,2:-2,:])).max()

simulation.set_initial_conditions()


plt.imshow(simulation.state.variables.salt[..., 0])
plt.show()

vm_T = nc.Dataset(r"acc.training.nc")
vm_S = nc.Dataset(r"acc.snapshot.nc")

print(vm_T.variables.keys())
print(vm_S.variables.keys())

vm_T.variables['salt']


vm.variables.keys()
plt.imshow(vm.variables['u'][:].data[-1,-1,:,:])
plt.show()