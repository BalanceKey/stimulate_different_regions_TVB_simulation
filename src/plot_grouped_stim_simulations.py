'''
Combine all simulations together and plot the results at the SEEG level
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import colorednoise as cn
import sys
sys.path.append('/Users/dollomab/MyProjects/Stimulation/VirtualEpilepsySurgery/VEP/core/')
import vep_prepare
roi = vep_prepare.read_vep_mrtrix_lut()

pid = 3
patients = {1: 'sub-603cf699f88f', 2: 'sub-2ed87927ff76', 3: 'sub-0c7ab65949e1',
            4: 'sub-9c71d0dbd98f', 5: 'sub-4b4606a742bd'}
stim_index = 344


data_path = f'../stim_different_roi_{patients[pid]}_stim{stim_index-2}-{stim_index}_4.npz'
data = np.load(data_path, allow_pickle=True)
tavg1 = data['tavg1']
tavg2 = data['tavg2']
tavg3 = data['tavg3']
time1 = data['time1']
time2 = data['time2']
time3 = data['time3']

# Group all three simulations together
tavg = np.concatenate((tavg1[:, 0, :, 0].T, tavg2[:, 0, :, 0].T, tavg3[:, 0, :, 0].T), axis=1)
time = np.concatenate((time1, time2 + time1[-1], time3 + time1[-1] + time2[-1]))

# Plot the results
scaleplt = 4
plt.figure(figsize=(10, 15), tight_layout=True)
for i in range(len(roi)):
    plt.plot(time, tavg[i, :] * scaleplt + i + 1, 'blue', linewidth=0.5)
plt.yticks(np.arange(len(roi)), roi)
# plt.xlim([time[0], time[-1]])
plt.ylim([-2, len(roi) + 1])
plt.show()

#%% Map the results to SEEG contacts
# Load SEEG electrodes
subject_dir = f'/Users/dollomab/MyProjects/Epinov_trial/patients/{patients[pid]}/vep'
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
plot = False
if plot:
    vep_prepare.plot_gain_matrix(bip_gain, bip_names, pid)
    plt.show()

seeg_signal = np.dot(bip_gain, tavg)  # map the source space activity to seeg space
#%% Plot the results at SEEG level
scaleplt = 0.5
plt.figure(figsize=(10, 15), tight_layout=True)
for i in range(len(bip_names)):
    plt.plot(time, (seeg_signal[i, :] - seeg_signal[i, 0]) * scaleplt + i + 1, 'blue', linewidth=0.5)
plt.yticks(np.arange(len(bip_names)), bip_names)
plt.xlim([time[0], time[-1]])
plt.ylim([-2, len(bip_names) + 1])
plt.title(f'SEEG signals - Bipolar montage - Patient {patients[pid]})')
plt.show()

# Applying a high pass filter to the signal to resseble AC recordings
def highpass_filter(y, sr):
    """In this case, the filter_stop_freq is that frequency below which the filter MUST act like a stop filter and filter_pass_freq is that frequency above which the filter MUST act like a pass filter.
       The frequencies between filter_stop_freq and filter_pass_freq are the transition region or band."""
    filter_stop_freq = 3  # Hz
    filter_pass_freq = 3  # Hz
    filter_order = 501
    # High-pass filter
    nyquist_rate = sr / 2.
    desired = (0, 0, 1, 1)
    bands = (0, filter_stop_freq, filter_pass_freq, nyquist_rate)
    filter_coefs = signal.firls(filter_order, bands, desired, nyq=nyquist_rate)
    # Apply high-pass filter
    filtered_audio = signal.filtfilt(filter_coefs, [1], y)
    return filtered_audio

y = highpass_filter(seeg_signal, 300)  # seeg
scaleplt = 0.5
plt.figure(figsize=(10, 15), tight_layout=True)
for i in range(len(bip_names)):
    plt.plot(time, (y[i, :] - y[i, 0]) * scaleplt + i + 1, 'blue', linewidth=0.5)
plt.yticks(np.arange(len(bip_names)), bip_names)
plt.xlim([time[0], time[-1]])
plt.ylim([-2, len(bip_names) + 1])
plt.title(f'SEEG signals - Bipolar montage - Patient {patients[pid]})')
plt.show()


# Plot only a subset of channels
# TODO add some noise here
ch_names = [ "B1-2", "B2-3", "B4-5", "B5-6", "B6-7",
        "B8-9", "TB1-2",  "TB4-5", "TB6-7",
         "A1-2", "A2-3", "A3-4", "A4-5", "A5-6", "A6-7",
         "A8-9", "A9-10", 
         "Im1-2", "Im3-4", "Im5-6", "Im6-7",
         "Ia1-2","Ia3-4", "Ia5-6", "Ia6-7",]

beta = 1  # the exponent
noise1 = cn.powerlaw_psd_gaussian(beta, y.shape)
beta = 2  # the exponent
noise2 = cn.powerlaw_psd_gaussian(beta, y.shape)
beta = 3  # the exponent
noise3 = cn.powerlaw_psd_gaussian(beta, y.shape)
y_new = y + noise1*0.2 + noise2 * 0.1

scaleplt = 0.2
plt.figure(figsize=(20, 20), tight_layout=True)
for i in range(len(ch_names)):
    ch_idx = bip_names.index(ch_names[i])
    # plt.plot(time, (y[ch_idx, :] - y[ch_idx, 0]) * scaleplt + i, 'blue', linewidth=1)
    plt.plot(time, (y_new[ch_idx, :] - y_new[ch_idx, 0]) * scaleplt + i, 'blue', linewidth=1)
plt.yticks(np.arange(len(ch_names)), ch_names, fontsize=26)
plt.xlim([time[0], time[-1]])
plt.ylim([-1, len(ch_names) + 1])
plt.title(f'Simulated timeseries', fontsize=50, fontweight='bold')
plt.xlabel('Time', fontsize=50)
plt.ylabel('Electrodes', fontsize=50)
plt.show()

