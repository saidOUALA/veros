import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
# from veros.setups.global_1deg_learning import GlobalOneDegreeLearningSetup
from veros.setups.acc_learning import ACCLearningSetup
from veros.io_tools.netcdf import extract_init_cond, load_timesteps_between

# setup learning simulation
simulation = ACCLearningSetup()
simulation.setup()

# extract variables needed to restart simulation
restart_vars = {var: meta for var, meta in simulation.state.var_meta.items() if meta.write_to_restart and meta.active}

# load (a chunk of) the training data
start_timestep = 10
end_timestep = 30
data_chunks = load_timesteps_between("acc.training.nc", start_timestep, end_timestep)

# construct a dictionary that contains the initial condition
idx_init = 1
initial_condition = extract_init_cond(data_chunks, restart_vars, idx=idx_init)

# setup the initial condition
simulation.set_initial_conditions_learning(simulation.state, initial_condition)

# run the simulation and extract the sequence
simulated_seq = simulation.run(extract_sequence=True, restart_vars=restart_vars)
simulated_seq_u = np.array(simulated_seq['u'])
GT_seq_u = data_chunks['u'][idx_init:idx_init+10+1]

idx_comp = -1
error = np.abs(
    simulated_seq_u[idx_comp] - GT_seq_u.filled(fill_value=0.0)[idx_comp].T)

# should be zero
plt.imshow(error[:, :, -1], cmap='jet')
plt.colorbar()
plt.show()