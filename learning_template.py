import numpy as np
from veros import runtime_settings
setattr(runtime_settings, 'backend', 'jax')
setattr(runtime_settings, 'device', 'gpu')
setattr(runtime_settings, 'force_overwrite', True)
setattr(runtime_settings, 'linear_solver', 'scipy_jax')
import netCDF4 as nc
import matplotlib.pyplot as plt
# from veros.setups.global_1deg_learning import GlobalOneDegreeLearningSetup
from veros.setups.acc_learning import ACCLearningSetup
from veros.io_tools.netcdf import extract_init_cond, load_timesteps_between
import random 




# path to training simulation:
path_training = "acc_runs/acc_simulation_quarter_post_spinup_GT_training_chunked/acc_simulation_quarter_post_spinup_GT_training_chunked"
# setup learning simulation
simulation = ACCLearningSetup()
simulation.setup()
n_steps = int(simulation.state.settings.runlen/simulation.state.settings.dt_mom)

# extract variables needed to restart simulation
restart_vars = {var: meta for var, meta in simulation.state.var_meta.items() if meta.write_to_restart and meta.active}

for i in range(100):
    random_list = random.sample(range(8), 6)
    for b in range(len(random_list)):
        print('batch : ', b)
        start_timestep = random_list[b]
        end_timestep = start_timestep + n_steps
        print('start_timestep : ', start_timestep)
        print('end_timestep : ', end_timestep)
        print('n_steps : ', n_steps)
        data_chunks = load_timesteps_between(path_training+".training.nc", start_timestep, end_timestep+2)# +2 is to adjust for the 2 initial conditions needed by the numerical scheme 

        # construct a dictionary that contains the initial condition
        idx_init = 1
        initial_condition = extract_init_cond(data_chunks, restart_vars, idx=idx_init)

        # setup the initial condition
        simulation.set_initial_conditions_learning(simulation.state, initial_condition)

        # run the simulation and extract the sequence
        simulated_seq = simulation.run(extract_sequence=True, restart_vars=restart_vars)
        
        # compute cost function (here error of horizontal velocities)
        simulated_seq_u = np.array(simulated_seq['u'])
        simulated_seq_v = np.array(simulated_seq['v'])
        
        print(idx_init)
        print(idx_init+n_steps+1)
        
        GT_seq_u = data_chunks['u'][idx_init:idx_init+n_steps+1]
        GT_seq_v = data_chunks['v'][idx_init:idx_init+n_steps+1]
        
        print("GT_seq_u.shape : ", GT_seq_u.shape)
        
        error_u = np.mean((np.swapaxes(simulated_seq_u,1,-1) - GT_seq_u.filled(fill_value=0.0))**2)
        error_v = np.mean((np.swapaxes(simulated_seq_v,1,-1) - GT_seq_v.filled(fill_value=0.0))**2)

        error = error_u+error_v

        
        print("error is : ", error)
        # modify this please ^^
        # compute here the gradients of the cost wrt r_bot 
        # error.backward()
        
        # gradient decent
        # simulation.xx.r_bot = simulation.xx.r_bot - lr*simulation.xx.r_bot.grad
        