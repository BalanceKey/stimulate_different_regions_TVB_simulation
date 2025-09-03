'''
Code to stimulate different ROIs within one simulation with TVB
'''

from tvb.simulator.lab import *
import matplotlib.pyplot as plt
import numpy as np
import time as tm
from plot_data import plot_pattern
import sys
sys.path.append('/Users/dollomab/MyProjects/Epileptor3D/epileptor3D_collab/src/')
from model4 import EpileptorStim
from my_integrator import HeunDeterministicAdapted, HeunStochasticAdapted
sys.path.append('/Users/dollomab/MyProjects/Stimulation/VirtualEpilepsySurgery/VEP/core/')
import vep_prepare
roi = vep_prepare.read_vep_mrtrix_lut()


patients = {1: 'sub-603cf699f88f', 2: 'sub-2ed87927ff76', 3: 'sub-0c7ab65949e1',
            4: 'sub-9c71d0dbd98f', 5: 'sub-4b4606a742bd'}
pid = 3
subject_dir = f'/Users/dollomab/MyProjects/Epinov_trial/patients/{patients[pid]}/vep'
# subject_dir = f'/Users/dollomab/MyProjects/Epinov_trial/stimulated_patients/{patients[pid]}/vep'


#%% Set up EZ and simulation length
EZ = ['Right-Amygdala', 'Right-Hippocampus-anterior']  # EZ for patient 3
simulation_length = 4000

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
epileptors.x0 = np.ones(n_regions)*(-2.2)  #x0_vector close to critical threshold
# epileptors.threshold = np.exp(-x0_vector) - 2.95 # TODO this is adjustable, how best to set it ?
epileptors.threshold = 20 / (1 + np.exp(10*(x0_vector + 2.1))) + 2 # sigmoid function centred around 2.1
assert np.all(epileptors.threshold > 0)  # check thresholds>0, otherwise seizure starts automatically

#%% Stimulus
# Stimulus computation - stimuli waveform as a bipolar biphasic pulse
# Stimulus parameters
# TODO sfreq here was chosen too small (128 Hz), how best to set it ?
sfreq = 128                         # how many steps there are in one second
onset = 2 * sfreq                                                 # stimulation onset
stim_length = 5 * sfreq + onset  # stimulation duration
T = 1 / 50 * sfreq                    # pulse repetition period [s]
tau = 1000/1000000 * sfreq               # pulse duration, number of steps [microsec]
I = 1                                 # pulse intensity [mA]
assert stim_length < simulation_length

class vector1D(equations.DiscreteEquation):
    equation = equations.Final(default="emp")
pulse1, _ = equations.PulseTrain(parameters={'T': T, 'tau': tau, 'amp': I, 'onset': onset}) \
    .get_series_data(max_range=stim_length, step=dt)
pulse2, _ = equations.PulseTrain(parameters={'T': T, 'tau': tau, 'amp': I, 'onset': onset + tau}) \
    .get_series_data(max_range=stim_length, step=dt)
pulse1_ts = [p[1] for p in pulse1]
pulse2_ts = [p[1] for p in pulse2]
pulse_ts = np.asarray(pulse1_ts) - np.asarray(pulse2_ts)
stimulus_ts = np.hstack((pulse_ts[:-1], np.zeros(int(np.ceil((simulation_length - stim_length) / dt)))))
eqn_t = vector1D()
eqn_t.parameters['emp'] = np.copy(stimulus_ts)

# Combining for spatiotemporal stimulus
stim_weights = np.zeros(len(roi))
stim_weights[ez_idx] = 1  # stimulate only EZ 1

plot = True

stimulus = patterns.StimuliRegion(temporal=eqn_t,
                                  connectivity=con,
                                  weight=stim_weights)



stimulus.configure_space()
stimulus.configure_time(np.arange(0., np.size(stimulus_ts), 1))
if plot:
    plot_pattern(stimulus, roi)


#%% Initial conditions
ic = [-1.4, -9.6, 2.97, 0.0]
#%% Simulator
sim = simulator.Simulator(model=epileptors,
                          initial_conditions=np.repeat(ic, len(roi)).reshape((1, len(ic), len(roi), 1)),
                          connectivity=con,
                          stimulus=stimulus,
                          coupling=coupl,
                          integrator=heunint,
                          conduction_speed=np.inf,
                          monitors=[mon_tavg])
sim.configure()

print("Starting simulation...")
tic = tm.time()
ttavg1 = sim.run(simulation_length=simulation_length/2)
print('Simulation ran for ', (tm.time() - tic) / 60.0, 'mins')


#%% Plot results
time, tavg = ttavg1[0]

plt.figure(figsize=(10, 15), tight_layout=True)
for i in range(len(roi)):
    plt.plot(time, tavg[:, 0, i, 0] + i + 1, 'blue', linewidth=0.5)
plt.yticks(np.arange(len(roi)), roi)
plt.xlim([time[0], time[-1]])
plt.ylim([-2, len(roi) + 1])
plt.show()

plt.figure(figsize=(10, 15), tight_layout=True)
for i in range(len(roi)):
    plt.plot(time, tavg[:, 3, i, 0] + i + 1, 'blue', linewidth=0.5)
plt.yticks(np.arange(len(roi)), roi)
plt.xlim([time[0], time[-1]])
plt.ylim([-2, len(roi) + 1])
plt.show()

max_stim_idx = stim_weights.argmax()
n_subplots = 4
f, axs = plt.subplots(n_subplots, 1, sharex='col')
for i in range(n_subplots):
    axs[i].plot(time, tavg[:, i, max_stim_idx, 0])
axs[3].axhline(sim.model.threshold[max_stim_idx], 0, time[-1])
plt.suptitle(f'{roi[max_stim_idx]}')
plt.show()



#%% TODO run a second simulation with different stimulus weights
# Combining for spatiotemporal stimulus
stim_weights = np.zeros(len(roi))
stim_weights[ez_idx[1]] = 1  # stimulate only EZ 2

plot = True

stimulus = patterns.StimuliRegion(temporal=eqn_t,
                                  connectivity=con,
                                  weight=stim_weights)

stimulus.configure_space()
stimulus.configure_time(np.arange(0., np.size(stimulus_ts), 1))
if plot:
    plot_pattern(stimulus, roi)

#%% Simulator
sim = simulator.Simulator(model=epileptors,
                          initial_conditions = tavg[-1][np.newaxis, :, :, :],  # continue from previous simulation
                          connectivity=con,
                          stimulus=stimulus,
                          coupling=coupl,
                          integrator=heunint,
                          conduction_speed=np.inf,
                          monitors=[mon_tavg])
sim.configure()

print("Starting simulation...")
tic = tm.time()
ttavg2 = sim.run(simulation_length=simulation_length)
print('Simulation ran for ', (tm.time() - tic) / 60.0, 'mins')

#%% Plot results
time, tavg = ttavg2[0]

plt.figure(figsize=(10, 15), tight_layout=True)
for i in range(len(roi)):
    plt.plot(time, tavg[:, 0, i, 0] + i + 1, 'blue', linewidth=0.5)
plt.yticks(np.arange(len(roi)), roi)
plt.xlim([time[0], time[-1]])
plt.ylim([-2, len(roi) + 1])
plt.show()

plt.figure(figsize=(10, 15), tight_layout=True)
for i in range(len(roi)):
    plt.plot(time, tavg[:, 3, i, 0] + i + 1, 'blue', linewidth=0.5)
plt.yticks(np.arange(len(roi)), roi)
plt.xlim([time[0], time[-1]])
plt.ylim([-2, len(roi) + 1])
plt.show()

max_stim_idx = stim_weights.argmax()
n_subplots = 4
f, axs = plt.subplots(n_subplots, 1, sharex='col')
for i in range(n_subplots):
    axs[i].plot(time, tavg[:, i, max_stim_idx, 0])
axs[3].axhline(sim.model.threshold[max_stim_idx], 0, time[-1])
plt.suptitle(f'{roi[max_stim_idx]}')
plt.show()

idx_roi = roi.index('Right-Amygdala')
n_subplots = 5
f, axs = plt.subplots(n_subplots, 1, sharex='col')
for i in range(n_subplots-1):
    axs[i].plot(time, tavg[:, i, idx_roi, 0])
axs[3].axhline(sim.model.threshold[idx_roi], 0, time[-1])
axs[4].plot(sim.stimulus.time[0], sim.stimulus.temporal_pattern[0] * sim.stimulus.spatial_pattern[idx_roi])
plt.suptitle(f'{roi[idx_roi]}')
plt.show()
