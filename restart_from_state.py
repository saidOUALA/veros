import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
#from veros.setups.global_1deg_learning import GlobalOneDegreeLearningSetup
from veros.setups.acc_learning import ACCLearningSetup
from veros.setups.acc import ACCSetup

simulation = ACCLearningSetup()
simulation.setup()

simulation_init = ACCSetup()
simulation_init.setup()

state = simulation.run(extract_state=True, iter = 2)



# extract variables needed to restart simulation
restart_vars = {var: meta for var, meta in simulation.state.var_meta.items() if meta.write_to_restart and meta.active}
restart_vars_init = {var: meta for var, meta in simulation_init.state.var_meta.items() if meta.write_to_restart and meta.active}

# construct a dictionary that contains those variables from the save simulation
vm_T = nc.Dataset(r"acc.training.nc")
training_simulation = {}
idx_training = 0
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
        print(key, vm_T.variables[key][:].filled(fill_value=0.0).shape)
        training_simulation[key] = vm_T.variables[key][:].filled(fill_value=0.0)[idx_training-1:idx_training+2].T
        training_simulation[key][..., -1] = 0*training_simulation[key][..., -1]

#simulation.set_initial_conditions_learning(simulation.state, training_simulation)
simulated_u = simulation.run(extract_sequence=True)

idx_comp = 1
error = np.abs(np.array(simulated_u)[idx_comp] - vm_T.variables['u'][:].filled(fill_value=0.0)[idx_training+idx_comp].T)

plt.imshow(error[:,:,-1],cmap = 'jet')
plt.colorbar()
plt.show()

vm_l = nc.Dataset(r"acc_learning.snapshot.nc")

np.abs(vm_l.variables['u'][:].filled(fill_value=0.0) - vm_T.variables['u'][:].filled(fill_value=0.0)[idx_training:idx_training+2,:,2:-2,2:-2]).max()
np.abs(vm_l.variables['u'][:].filled(fill_value=0.0)[-1].T - simulation.state.variables.u[2:-2,2:-2,:,simulation.state.variables.tau]).max()


vm_T.variables['u'][:][idx_training+1:]


np.abs(simulation.state.variables.v[2:-2,2:-2,:,simulation.state.variables.tau] - vm_T.variables['v'][:].filled(fill_value=0.0)[idx_training+3].T[2:-2,2:-2,:]).max()
np.abs(simulation.state.variables.v[2:-2,2:-2,:,simulation.state.variables.tau] - vm_T.variables['v'][:].filled(fill_value=0.0)[idx_training+2].T[2:-2,2:-2,:]).max()



var_data = variables.remove_ghosts(var_data, var.dims)[tmask].T
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