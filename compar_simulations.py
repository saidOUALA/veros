import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
#from veros.setups.global_1deg_learning import GlobalOneDegreeLearningSetup
#from veros.setups.acc_learning import ACCLearningSetup
from veros.setups.acc import ACCSetup
from veros.io_tools.netcdf import extract_init_cond, load_timesteps_between

#simulation = ACCSetup()
#simulation.setup()
#simulation.run()
#check stream function
start_timestep = 0
end_timestep = 10
data_chunks_GT = load_timesteps_between("acc_simulation_GT/acc_post_spinup_GT_config.training.nc", start_timestep, end_timestep)
plt.imshow(data_chunks_GT['psi'][0,::-1,:][2:-2, 2:-2]/1E6, cmap = 'jet', vmin =-50, vmax = 160)
plt.colorbar()
plt.show()
# diagnostic spinup:
acc_diag_spinup1 = nc.Dataset(r"acc_restart_from_test_diag.energy.nc")
acc_diag_spinup2 = nc.Dataset(r"acc_restart_from_50000_test_diag.energy.nc")

acc_diag_spinup1flux = nc.Dataset(r"acc_restart_from_test_diag.acc_diags.nc")
acc_diag_spinup2flux = nc.Dataset(r"acc_restart_from_50000_test_diag.acc_diags.nc")

km = np.concatenate((acc_diag_spinup1.variables['k_m'][:],acc_diag_spinup2.variables['k_m'][:]))
flux = np.concatenate((acc_diag_spinup1flux.variables['flux_north_south'][:],acc_diag_spinup2flux.variables['flux_north_south'][:]))
Time = np.concatenate((acc_diag_spinup1.variables['Time'][:],acc_diag_spinup2.variables['Time'][:]))

plt.plot(flux)
plt.xlabel('time (timesteps)')
plt.ylabel('Volumetric flow rate (m3/s)')
plt.show()

plt.plot(Time,km)
plt.xlabel('time (days)')
plt.ylabel('Mean kinetic energy')
plt.show()

plt.plot(Time,km)
plt.xlabel('time (days)')
plt.ylabel('Mean kinetic energy')
plt.show()
# diagnostic RBOT:

###################################################""diag rbot##########################################################
# ACC diag flux:
file_path = r"/media/administrateur/B612AA5912AA1E7D/veros_runs/acc/"
acc_diag = nc.Dataset(r"acc_simulation_GT/acc_post_spinup_GT_config.acc_diags.nc")
acc_diag_rbot = nc.Dataset(r"acc_simulation_rbot/acc_post_spinup_rbot_config_times_10.acc_diags.nc")
acc_diag_rbotd10 = nc.Dataset(file_path+"acc_simulation_rbot_d10/acc_post_spinup_rbot_config_per_10.acc_diags.nc")

# plot
flux_rbotd10 = acc_diag_rbotd10.variables['flux_north_south'][:]
fluxGT = np.concatenate((flux,acc_diag.variables['flux_north_south'][:]))
fluxmodif_Rbot = np.concatenate((flux,acc_diag_rbot.variables['flux_north_south'][:]))
fluxmodif_Rbotd10 = np.concatenate((flux,flux_rbotd10))


plt.plot(fluxmodif_Rbot)
plt.plot(fluxmodif_Rbotd10)
plt.plot(fluxGT, c='r')
plt.show()

plt.plot(acc_diag_rbotd10.variables['flux_north_south'][:200])
plt.plot(acc_diag_rbot.variables['flux_north_south'][:200])
plt.plot(acc_diag.variables['flux_north_south'][:200],c = 'r')
plt.show()

# imshow
end_time = 2000
time_span = range(end_time)
norm_fact = 1
[X,Y]=np.meshgrid(time_span, acc_diag.variables['xu'])
gtminusrbot = acc_diag.variables['flux_north_south'][:end_time] - acc_diag_rbot.variables['flux_north_south'][:end_time]
plt.figure(figsize=(7,10))
plt.subplot(3,1,1);plt.pcolor(X,Y,acc_diag.variables['flux_north_south'][:end_time].T/norm_fact);plt.ylabel('xu');plt.xlabel('time');plt.title('r_bot_init_1E-5')
plt.colorbar()
plt.subplot(3,1,2);plt.pcolor(X,Y,acc_diag_rbot.variables['flux_north_south'][:end_time].T/norm_fact);plt.clim([1.0E8, 1.8E8]);plt.ylabel('xu');plt.xlabel('time');plt.title('r_bot_1E-4')
plt.colorbar()
plt.subplot(3,1,3);plt.pcolor(X,Y,gtminusrbot.T/norm_fact);plt.ylabel('xu');plt.xlabel('time');plt.title('difference')
plt.colorbar()
plt.tight_layout()
#plt.subplot(2,2,1);plt.pcolor(X,Y,acc_diag.variables['flux_north_south'][:end_time]);#clim([-10,10]);ylabel('Lorenz-96 times');title('Local analog data assimilation')
plt.show()

# ACC diag zonal mean:
idx_plt = 3
[X,Y]=np.meshgrid(acc_diag.variables['yt'], acc_diag.variables['zt'])
gtminusrbot = acc_diag_rbot.variables['temp_zonal_mean'][:][idx_plt]-acc_diag.variables['temp_zonal_mean'][:][idx_plt]
plt.figure(figsize=(7,10))
plt.subplot(3,1,1);plt.pcolor(X,Y,acc_diag.variables['temp_zonal_mean'][:][idx_plt],cmap ='jet');plt.ylabel('zt');plt.xlabel('yt');plt.title('zonal mean (temperature, rbot 1E-5)')
plt.colorbar()
plt.subplot(3,1,2);plt.pcolor(X,Y,acc_diag_rbot.variables['temp_zonal_mean'][:][idx_plt],cmap ='jet');plt.ylabel('zt');plt.xlabel('yt');plt.title('zonal mean (temperature, rbot 1E-4)')
plt.colorbar()
plt.subplot(3,1,3);plt.pcolor(X,Y,gtminusrbot,cmap ='jet');plt.ylabel('zt');plt.xlabel('yt');plt.title('error')
plt.colorbar()
plt.show()

idx_plt = -1
[X,Y]=np.meshgrid(acc_diag.variables['yt'], acc_diag.variables['zt'])
gtminusrbot = acc_diag_rbotd10.variables['temp_zonal_mean'][:][idx_plt]-acc_diag.variables['temp_zonal_mean'][:][idx_plt]
plt.figure(figsize=(7,10))
plt.subplot(3,1,1);plt.pcolor(X,Y,acc_diag.variables['temp_zonal_mean'][:][idx_plt],cmap ='jet');plt.ylabel('zt');plt.xlabel('yt');plt.title('zonal mean (temperature, rbot 1E-5)')
plt.colorbar()
plt.subplot(3,1,2);plt.pcolor(X,Y,acc_diag_rbotd10.variables['temp_zonal_mean'][:][idx_plt],cmap ='jet');plt.ylabel('zt');plt.xlabel('yt');plt.title('zonal mean (temperature, rbot 1E-4)')
plt.colorbar()
plt.subplot(3,1,3);plt.pcolor(X,Y,gtminusrbot,cmap ='jet');plt.ylabel('zt');plt.xlabel('yt');plt.title('error')
plt.colorbar()
plt.show()


idx_plt = -1
[X,Y]=np.meshgrid(acc_diag.variables['yt'], acc_diag.variables['zt'])
gtminusrbot = acc_diag_rbot.variables['salt_zonal_mean'][:][idx_plt]-acc_diag.variables['salt_zonal_mean'][:][idx_plt]
plt.figure(figsize=(7,10))
plt.subplot(3,1,1);plt.pcolor(X,Y,acc_diag.variables['salt_zonal_mean'][:][idx_plt],cmap ='jet');plt.clim([]);plt.ylabel('zt');plt.xlabel('yt');plt.title('zonal mean (salt, rbot 1E-5)')
plt.colorbar()
plt.subplot(3,1,2);plt.pcolor(X,Y,acc_diag_rbot.variables['salt_zonal_mean'][:][idx_plt],cmap ='jet');plt.ylabel('zt');plt.xlabel('yt');plt.title('zonal mean (salt, rbot 1E-4)')
plt.colorbar()
plt.subplot(3,1,3);plt.pcolor(X,Y,gtminusrbot,cmap ='jet');plt.ylabel('zt');plt.xlabel('yt');plt.title('error')
plt.colorbar()
plt.show()


idx_plt = -1
[X,Y]=np.meshgrid(acc_diag.variables['yt'], acc_diag.variables['zt'])
gtminusrbot = acc_diag_rbotd10.variables['salt_zonal_mean'][:][idx_plt]-acc_diag.variables['salt_zonal_mean'][:][idx_plt]
plt.figure(figsize=(7,10))
plt.subplot(3,1,1);plt.pcolor(X,Y,acc_diag.variables['salt_zonal_mean'][:][idx_plt],cmap ='jet');plt.ylabel('zt');plt.xlabel('yt');plt.title('zonal mean (salt, rbot 1E-5)')
plt.colorbar()
plt.subplot(3,1,2);plt.pcolor(X,Y,acc_diag_rbotd10.variables['salt_zonal_mean'][:][idx_plt],cmap ='jet');plt.ylabel('zt');plt.xlabel('yt');plt.title('zonal mean (salt, rbot 1E-4)')
plt.colorbar()
plt.subplot(3,1,3);plt.pcolor(X,Y,gtminusrbot,cmap ='jet');plt.ylabel('zt');plt.xlabel('yt');plt.title('error')
plt.colorbar()
plt.show()


# check difference state vars
# load a chunk of training data:
# load (a chunk of) the training data
start_timestep = 0
end_timestep = 100
data_chunks_GT = load_timesteps_between("acc_simulation_GT/acc_post_spinup_GT_config.training.nc", start_timestep, end_timestep)
data_chunks_diag = load_timesteps_between("acc_simulation_rbot/acc_post_spinup_rbot_config_times_10.training.nc", start_timestep, end_timestep)
data_chunks_diagd10 = load_timesteps_between(file_path+"acc_simulation_rbot_d10/acc_post_spinup_rbot_config_per_10.training.nc", start_timestep, end_timestep)
mse_u = ((data_chunks_GT['u']-data_chunks_diag['u'])**2).mean(axis=(1, 2, 3))
mse_v = ((data_chunks_GT['v']-data_chunks_diag['v'])**2).mean(axis=(1, 2, 3))
mse_w = ((data_chunks_GT['w']-data_chunks_diag['w'])**2).mean(axis=(1, 2, 3))
mse_salt = ((data_chunks_GT['salt']-data_chunks_diag['salt'])**2).mean(axis=(1, 2, 3))
mse_temp = ((data_chunks_GT['temp']-data_chunks_diag['temp'])**2).mean(axis=(1, 2, 3))

mse_u_rbot_d10 = ((data_chunks_GT['u']-data_chunks_diagd10['u'])**2).mean(axis=(1, 2, 3))
mse_v_rbot_d10 = ((data_chunks_GT['v']-data_chunks_diagd10['v'])**2).mean(axis=(1, 2, 3))
mse_w_rbot_d10 = ((data_chunks_GT['w']-data_chunks_diagd10['w'])**2).mean(axis=(1, 2, 3))
mse_salt_rbot_d10 = ((data_chunks_GT['salt']-data_chunks_diagd10['salt'])**2).mean(axis=(1, 2, 3))
mse_temp_rbot_d10 = ((data_chunks_GT['temp']-data_chunks_diagd10['temp'])**2).mean(axis=(1, 2, 3))


plt.imshow(data_chunks_GT['psi'][-1,::-1,:][2:-2, 2:-2]/1E6 -data_chunks_diag['psi'][-1,::-1,:][2:-2, 2:-2]/1E6, cmap = 'jet')
plt.colorbar()
plt.show()

plt.imshow(data_chunks_diag['psi'][-1,::-1,:][2:-2, 2:-2]/1E6, cmap = 'jet', vmin =-50, vmax = 160)
plt.colorbar()
plt.show()

plt.plot(mse_u, label = 'mse u')
plt.plot(mse_v, label = 'mse v')
plt.plot(mse_w, label = 'mse w')
plt.plot(mse_salt, label = 'mse salt')
plt.plot(mse_temp, label = 'mse temp')
plt.legend()
plt.show()

plt.plot(mse_u[:20], label = 'mse u')
plt.plot(mse_v[:20], label = 'mse v')
plt.plot(mse_w[:20], label = 'mse w')
plt.plot(mse_salt[:20], label = 'mse salt')
plt.plot(mse_temp[:20], label = 'mse temp')
plt.legend()
plt.show()
################################################### test with A_h ##############################################
acc_diag_ah = nc.Dataset(file_path+"acc_simulation_Ah_x10/acc_post_spinup_Ah_config_x_10.acc_diags.nc")
acc_diag_ahd10 = nc.Dataset(file_path+"acc_simulation_Ah_d10/acc_post_spinup_Ah_config_per_10.acc_diags.nc")

# plot
flux_ah_d10 = acc_diag_rbotd10.variables['flux_north_south'][:]
fluxmodif_ah = np.concatenate((flux,acc_diag_ah.variables['flux_north_south'][:]))
fluxmodif_ah_d10 = np.concatenate((flux,flux_ah_d10))


#plt.plot(fluxmodif_ah)
plt.plot(fluxmodif_ah_d10)
plt.plot(fluxGT, c='r')
plt.show()

plt.plot(acc_diag_ahd10.variables['flux_north_south'][:20])
plt.plot(acc_diag_ah.variables['flux_north_south'][:20])
plt.plot(acc_diag.variables['flux_north_south'][:20],c = 'r')
plt.show()


idx_plt = -1
[X,Y]=np.meshgrid(acc_diag.variables['yt'], acc_diag.variables['zt'])
gtminusrbot = acc_diag_ahd10.variables['temp_zonal_mean'][:][idx_plt]-acc_diag.variables['temp_zonal_mean'][:][idx_plt]
plt.figure(figsize=(7,10))
plt.subplot(3,1,1);plt.pcolor(X,Y,acc_diag.variables['temp_zonal_mean'][:][idx_plt],cmap ='jet');plt.ylabel('zt');plt.xlabel('yt');plt.title('zonal mean (temperature, rbot 1E-5)')
plt.colorbar()
plt.subplot(3,1,2);plt.pcolor(X,Y,acc_diag_ahd10.variables['temp_zonal_mean'][:][idx_plt],cmap ='jet');plt.ylabel('zt');plt.xlabel('yt');plt.title('zonal mean (temperature, rbot 1E-4)')
plt.colorbar()
plt.subplot(3,1,3);plt.pcolor(X,Y,gtminusrbot,cmap ='jet');plt.ylabel('zt');plt.xlabel('yt');plt.title('error')
plt.colorbar()
plt.show()