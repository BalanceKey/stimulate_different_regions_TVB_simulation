'''
Code to stimulate different ROIs within one simulation with TVB
'''

from tvb.simulator.lab import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time as tm
import re
import sys
from run_simulation import run_simulation
from epistim_model import EpileptorStim
from my_integrator import HeunDeterministicAdapted, HeunStochasticAdapted
sys.path.append('/Users/dollomab/MyProjects/Stimulation/VirtualEpilepsySurgery/VEP/core/')
import vep_prepare
roi = vep_prepare.read_vep_mrtrix_lut()

pid = 3
plot = False
# simulation_length = 4000 # Here we will define it as stim_length + 4 secs after stim

patients = {1: 'sub-603cf699f88f', 2: 'sub-2ed87927ff76', 3: 'sub-0c7ab65949e1',
            4: 'sub-9c71d0dbd98f', 5: 'sub-4b4606a742bd'}
subject_dir = f'/Users/dollomab/MyProjects/Epinov_trial/patients/{patients[pid]}/vep'
# subject_dir = f'/Users/dollomab/MyProjects/Epinov_trial/stimulated_patients/{patients[pid]}/vep'
stimulation_data = f'~/MyProjects/Stimulation/Project_DBS_neural_fields/stimulation_data_manager_{patients[pid]}.csv'

# %% Load stimulation parameters
df = pd.read_csv(stimulation_data)
iterable_params = []
for i in range(0, df.shape[0]):
    stim_index = i  # taking each stimulation one by one
    channels = re.findall(r'[a-zA-Z\']+', df['stim_electrodes'][stim_index])
    channel_nr = re.findall(r'[0-9]+', df['stim_electrodes'][stim_index])

    if channels[0][0] == 'I':
        channels = [channels[0][0] + channels[0][1].lower(), channels[1][0] + channels[1][1].lower()]
    if patients[pid] == 'sub-4b4606a742bd' or patients[pid] == 'sub-2ed87927ff76' or patients[pid] == 'sub-0c7ab65949e1':
        if channels[0] == 'LES':
            channels = [channels[0][0] + channels[0][1:].lower(), channels[1][0] + channels[1][1:].lower()]
        if channels[0] == 'LESA' or channels[0] == 'LESB':
            channels = [channels[0][0] + channels[0][1:3].lower() + channels[0][-1],
                        channels[1][0] + channels[1][1:3].lower() + channels[0][-1]]
        if int(channel_nr[0]) == int(channel_nr[1]) + 1:  # e.g. channel A3-2 instead of A2-3
            channel_nr = [channel_nr[1], channel_nr[0]]  # quick fix: inverting channel numbers here

    if patients[pid] == 'sub-603cf699f88f' or patients[pid]  == 'sub-9c71d0dbd98f' or patients[pid] == 'sub-4b4606a742bd':
        stimulation_parameters = {'choi': channels[0] + channel_nr[0] + '-' + channel_nr[1],
                                'freq': float(df['frequency'][stim_index].replace(',', '.')),  # Hz
                                'amp': float(df['intensity'][stim_index].replace(',', '.')),  # mA
                                'duration': float(df['duration'][stim_index]),  # seconds
                                'tau': float(df['pulse_width'][stim_index]),  # microseconds
                                'sfreq': 512,  # Hz
                                'stim_index': stim_index
                                }
    else:
        stimulation_parameters = {'choi': channels[0] + channel_nr[0] + '-' + channel_nr[1],
                                'freq': float(df['frequency'][stim_index]),  # Hz
                                'amp': float(df['intensity'][stim_index]),  # mA
                                'duration': float(df['duration'][stim_index]),  # seconds
                                'tau': float(df['pulse_width'][stim_index]),  # microseconds
                                'sfreq': 512,  # Hz
                                'stim_index': stim_index
                                }
    iterable_params.append(stimulation_parameters)

#%% Load SEEG electrodes
seeg_xyz = vep_prepare.read_seeg_xyz(subject_dir)                # read seeg electrodes information
seeg_xyz_names = [channel_name for channel_name, _ in seeg_xyz]  # read monopolar electrode names
#%% Load gain matrix
gain = np.loadtxt(f'{subject_dir}/elec/gain_inv-square.vep.txt')  # import gain matrix
assert len(seeg_xyz_names) == gain.shape[0]
bip_gain, bip_xyz, bip_names = vep_prepare.bipolarize_gain_minus(gain, seeg_xyz, seeg_xyz_names)  # bipolarize gain
bip_gain_prior, _, _ = vep_prepare.bipolarize_gain_minus(gain, seeg_xyz, seeg_xyz_names, is_minus=False)
# Remove cereberall cortex influence
bip_gain[:, roi.index('Left-Cerebellar-cortex')] = bip_gain.min()
bip_gain[:, roi.index('Right-Cerebellar-cortex')] = bip_gain.min()
bip_gain_prior[:, roi.index('Left-Cerebellar-cortex')] = bip_gain_prior.min()
bip_gain_prior[:, roi.index('Right-Cerebellar-cortex')] = bip_gain_prior.min()

# Normalize gain matrix
bip_gain_norm = (bip_gain - bip_gain.min()) / (bip_gain.max() - bip_gain.min())
bip_gain_prior_norm = (bip_gain_prior - bip_gain_prior.min()) / (bip_gain_prior.max() - bip_gain_prior.min())
if plot:
    vep_prepare.plot_gain_matrix(bip_gain, bip_names, pid)
    plt.show()

#%% Set up EZ and simulation length
EZ = ['Right-Amygdala', 'Right-Hippocampus-anterior']  # EZ for patient 3

#%% Coupling
coupling_factor = 0.5
coupl = coupling.Difference(a=np.array([coupling_factor]))

#%% Integrator - deterministic
dt = 0.02
heunint = HeunDeterministicAdapted(dt=dt)

#%% Monitors
mon_tavg = monitors.TemporalAverage(period=0.9765625)
# period = 1 / sfreq * 1000  # period is in ms, e.g. 512 Hz => 1.953125 ms

#%%  Global connectivity
con = connectivity.Connectivity.from_file(f'{subject_dir}/tvb/connectivity.vep.zip')
con.tract_lengths = np.zeros((con.tract_lengths.shape))  # no time-delays
con.weights[np.diag_indices(con.weights.shape[0])] = 0
con.weights /= con.weights.max()
con.configure()
assert con.number_of_regions == 162
n_regions = con.number_of_regions

#%% Set up Epileptor model parameters
epileptors = EpileptorStim(variables_of_interest=['x1', 'y1', 'z', 'm'])
epileptors.r = np.array([0.0005])
epileptors.r2 = np.ones(n_regions) * (0.002)#(0.0011)
epileptors.Istim = np.ones(n_regions) * (0.)
epileptors.Ks = np.ones(n_regions) * (-3)
epileptors.Kf = np.ones(n_regions) * (-0.22)  # ?
epileptors.Kvf = np.ones(n_regions) * (-0.085)  # ?
# epileptors.threshold = np.ones(len(roi)) * (10.)
# Set up EZ network: here we convert x0 - > threshold by using the formula
# threshold = exp^(-x0) - 2  (x0 values close to threshold for seizing)
x0_vector = np.ones(len(roi)) * -2.5
ez_idx = [roi.index(ez) for ez in EZ]
x0_vector[ez_idx] = -1.65
epileptors.x0 = np.ones(n_regions)*(-2.3)  #x0_vector close to critical threshold
# epileptors.threshold = np.exp(-x0_vector) - 2.95 # TODO this is adjustable, how best to set it ?
epileptors.threshold = 20 / (1 + np.exp(10*(x0_vector + 2.1))) + 1.13 # sigmoid function centred around 2.1
assert np.all(epileptors.threshold > 0)  # check thresholds>0, otherwise seizure starts automatically

# %% Choose stimulation parameters and run first simulation
stim_index = 342 # choose which stimulation to run
stimulation_parameters = iterable_params[stim_index]  # take the first stimulation parameters set
# Initial conditions
ic = [-1.4, -9.6, 2.97, 0.0]
init_conditions = np.repeat(ic, len(roi)).reshape((1, len(ic), len(roi), 1))
ttavg1 = run_simulation(stimulation_parameters, init_conditions, dt, epileptors, con, coupl, heunint, 
                        mon_tavg, bip_names, bip_gain_prior_norm, roi, pre_stim_duration=8, plot=False)
#%% Plot results
time, tavg = ttavg1[0]

if plot:
    plt.figure(figsize=(10, 15), tight_layout=True)
    for i in range(len(roi)):
        plt.plot(time, tavg[:, 0, i, 0] + i + 1, 'blue', linewidth=0.5)
    plt.yticks(np.arange(len(roi)), roi)
    plt.xlim([time[0], time[-1]])
    plt.ylim([-2, len(roi) + 1])
    plt.show()

    stim_weights = bip_gain_prior_norm[bip_names.index(stimulation_parameters['choi'])]
    max_stim_idx = stim_weights.argmax()
    n_subplots = 4
    f, axs = plt.subplots(n_subplots, 1, sharex='col')
    for i in range(n_subplots):
        axs[i].plot(time, tavg[:, i, max_stim_idx, 0])
    # axs[3].axhline(epileptors.threshold[max_stim_idx], 0, time[-1])
    plt.suptitle(f'{roi[max_stim_idx]}')
    plt.show()

    idx_roi = roi.index('Right-Hippocampus-anterior')
    n_subplots = 4
    f, axs = plt.subplots(n_subplots, 1, sharex='col')
    for i in range(n_subplots):
        axs[i].plot(time, tavg[:, i, idx_roi, 0])
    # axs[3].axhline(sim.model.threshold[idx_roi], 0, time[-1])
    # axs[4].plot(sim.stimulus.time[0], sim.stimulus.temporal_pattern[0] * sim.stimulus.spatial_pattern[idx_roi])
    plt.suptitle(f'{roi[idx_roi]}')
    plt.show()

#%% Second stimulus computation - Run a second simulation with different stimulus weights
stim_index = 342 + 1 # choose which stimulation to run
stimulation_parameters = iterable_params[stim_index]  # take the first stimulation parameters set
init_conditions=tavg[-1][np.newaxis, :, :, :]  # continue from previous simulation
ttavg2 = run_simulation(stimulation_parameters, init_conditions, dt, epileptors, con, coupl, heunint, 
                        mon_tavg, bip_names, bip_gain_prior_norm, roi, plot=False)
#%% Plot results
time, tavg = ttavg2[0]

if plot:
    plt.figure(figsize=(10, 15), tight_layout=True)
    for i in range(len(roi)):
        plt.plot(time, tavg[:, 0, i, 0] + i + 1, 'blue', linewidth=0.5)
    plt.yticks(np.arange(len(roi)), roi)
    plt.xlim([time[0], time[-1]])
    plt.ylim([-2, len(roi) + 1])
    plt.show()

    stim_weights = bip_gain_prior_norm[bip_names.index(stimulation_parameters['choi'])]
    max_stim_idx = stim_weights.argmax()
    n_subplots = 4
    f, axs = plt.subplots(n_subplots, 1, sharex='col')
    for i in range(n_subplots):
        axs[i].plot(time, tavg[:, i, max_stim_idx, 0])
    # axs[3].axhline(epileptors.threshold[max_stim_idx], 0, time[-1])
    plt.suptitle(f'{roi[max_stim_idx]}')
    plt.show()

    idx_roi = roi.index('Right-Hippocampus-anterior')
    n_subplots = 4
    f, axs = plt.subplots(n_subplots, 1, sharex='col')
    for i in range(n_subplots):
        axs[i].plot(time, tavg[:, i, idx_roi, 0])
    # axs[3].axhline(sim.model.threshold[idx_roi], 0, time[-1])
    # axs[4].plot(sim.stimulus.time[0], sim.stimulus.temporal_pattern[0] * sim.stimulus.spatial_pattern[idx_roi])
    plt.suptitle(f'{roi[idx_roi]}')
    plt.show()

#%% Third stimulation and last one
stim_index = 342 + 2 # choose which stimulation to run
stimulation_parameters = iterable_params[stim_index]  # take the first stimulation parameters set
init_conditions=tavg[-1][np.newaxis, :, :, :]  # continue from previous simulation
ttavg3 = run_simulation(stimulation_parameters, init_conditions, dt, epileptors, con, coupl, heunint, 
                        mon_tavg, bip_names, bip_gain_prior_norm, roi, post_stim_duration=30, plot=False)

#%% Plot results
time, tavg = ttavg3[0]

if plot:
    plt.figure(figsize=(10, 15), tight_layout=True)
    for i in range(len(roi)):
        plt.plot(time, tavg[:, 0, i, 0] + i + 1, 'blue', linewidth=0.5)
    plt.yticks(np.arange(len(roi)), roi)
    plt.xlim([time[0], time[-1]])
    plt.ylim([-2, len(roi) + 1])
    plt.show()

    stim_weights = bip_gain_prior_norm[bip_names.index(stimulation_parameters['choi'])]
    max_stim_idx = stim_weights.argmax()
    n_subplots = 4
    f, axs = plt.subplots(n_subplots, 1, sharex='col')
    for i in range(n_subplots):
        axs[i].plot(time, tavg[:, i, max_stim_idx, 0])
    # axs[3].axhline(sim.model.threshold[max_stim_idx], 0, time[-1])
    plt.suptitle(f'{roi[max_stim_idx]}')
    plt.show()

    # idx_roi = roi.index('Right-Hippocampus-anterior')
    idx_roi = roi.index('Right-Amygdala')
    n_subplots = 4
    f, axs = plt.subplots(n_subplots, 1, sharex='col')
    for i in range(n_subplots):
        axs[i].plot(time, tavg[:, i, idx_roi, 0])
    axs[3].axhline(epileptors.threshold[idx_roi], 0, time[-1])
    axs[3].set_ylim([0, 1.7])
    # axs[4].plot(sim.stimulus.time[0], sim.stimulus.temporal_pattern[0] * sim.stimulus.spatial_pattern[idx_roi])
    plt.suptitle(f'{roi[idx_roi]}, maximum m = {tavg[:, 3, idx_roi, 0].max():.2f}')
    plt.show()

# #%% TODO combine all three simulations together and plot the entire timeseries
time1, tavg1 = ttavg1[0]
time2, tavg2 = ttavg2[0]
time3, tavg3 = ttavg3[0]

time = np.concatenate((time1, time2 + time1[-1], time3 + time1[-1] + time2[-1]))
xtavg = np.concatenate((tavg1[:, 0, :, 0].T, tavg2[:, 0, :, 0].T, tavg3[:, 0, :, 0].T), axis=1)
ytavg = np.concatenate((tavg1[:, 1, :, 0].T, tavg2[:, 1, :, 0].T, tavg3[:, 1, :, 0].T), axis=1)
mtavg = np.concatenate((tavg1[:, 3, :, 0].T, tavg2[:, 3, :, 0].T, tavg3[:, 3, :, 0].T), axis=1)
ztavg = np.concatenate((tavg1[:, 2, :, 0].T, tavg2[:, 2, :, 0].T, tavg3[:, 2, :, 0].T), axis=1)
idx_roi = roi.index('Right-Amygdala')
n_subplots = 3
f, axs = plt.subplots(n_subplots, 1, sharex='col', figsize=(20, 5))
axs[0].plot(time, xtavg[idx_roi], color='purple', linewidth=2)
# axs[1].plot(time, ytavg[idx_roi], color='orange')
axs[1].plot(time, ztavg[idx_roi], color='purple', linewidth=2)
axs[2].plot(time, mtavg[idx_roi], color='purple', linewidth=2)
axs[2].axhline(epileptors.threshold[idx_roi], 0, time[-1], color='black', linestyle='--')
axs[2].set_ylim([0, 1.7])
axs[0].set_ylabel('x1', fontsize=25)
axs[1].set_ylabel('z', fontsize=25)
axs[2].set_ylabel('m', fontsize=25)
axs[2].set_xlabel('Time (ms)', fontsize=25)
# axs[4].plot(sim.stimulus.time[0], sim.stimulus.temporal_pattern[0] * sim.stimulus.spatial_pattern[idx_roi])
plt.suptitle(f'{roi[idx_roi]} simulation', fontsize=25)
plt.tight_layout()
plt.show()


# Save all these simulation results
save = False
if save:
    np.savez(f'../stim_different_roi_{patients[pid]}_stim{stim_index-2}-{stim_index}_3.npz', 
            time1=ttavg1[0][0], tavg1=ttavg1[0][1], 
            time2=ttavg2[0][0], tavg2=ttavg2[0][1],
            time3=ttavg3[0][0], tavg3=ttavg3[0][1])




