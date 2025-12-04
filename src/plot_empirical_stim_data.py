import numpy as np
import matplotlib.pyplot as plt
import mne
import os.path as op
import sys
sys.path.append('/Users/dollomab/MyProjects/Stimulation/VirtualEpilepsySurgery/VEP/core/')
import vep_prepare
roi = vep_prepare.read_vep_mrtrix_lut()
from plot_data import plot_SEEG_timeseries

subject_dir = '/Users/dollomab/MyProjects/Epinov_trial/stimulated_patients/sub-0c7ab65949e1/vep/'
vhdr_file = subject_dir + '../ieeg/sub-0c7ab65949e1_ses-01_task-stim_run-04_ieeg.vhdr'
# raw.info['line_freq'] = 50  # specify power line frequency to enable notch filtering
# raw.filter(1., 150.)  # bandpass filter
# raw.notch_filter(np.arange(50, 151, 50))  # notch filter
# raw.resample(250.)  # resample to 250Hz

# %% Read stim data
bad_contacts = []
remove_cerebellar = True
raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)
raw._data *= 1e6
fname_bad = f'{vhdr_file}.bad'                               # read the bad channel
if op.exists(fname_bad):
    raw.load_bad_channels(fname_bad)
else:
    print('>> Warning: bad channels file', fname_bad, 'not found => assuming no bad channels')
if bad_contacts:
    print('>> Extending bad contacts with ' + ', '.join(bad_contacts))
    raw.info['bads'].extend(bad_contacts)
new_names = [name.replace(" ", "") for name in raw.ch_names] # get rid of possible spaces in the names
new_names = [name.replace(".", "") for name in new_names]
raw.rename_channels(dict(zip(raw.ch_names, new_names)))
seeg_xyz = vep_prepare.read_seeg_xyz(subject_dir)           # read from GARDEL file
seeg_xyz_names = [label for label, _ in seeg_xyz]
raw = raw.pick_types(meg=False, eeg=True, exclude="bads")
try:
    raw = raw.pick(seeg_xyz_names) # check wether all GARDEL channels exist in the raw seeg file
except:
    available_chs = set(raw.ch_names)
    seeg_xyz_names_present = [ch for ch in seeg_xyz_names if ch in available_chs]
    missing_chs = [ch for ch in seeg_xyz_names if ch not in available_chs]
    if missing_chs:
        print('Warning: missing GARDEL channels, skipping:', missing_chs)
    raw = raw.pick(seeg_xyz_names_present, verbose=True)
inv_gain_file = f'{subject_dir}/elec/gain_inv-square.vep.txt'       # read gain
invgain = np.loadtxt(inv_gain_file)
bip_gain_inv_minus, bip_xyz, bip_name = vep_prepare.bipolarize_gain_minus(invgain, seeg_xyz, seeg_xyz_names)
bip = vep_prepare._bipify_raw(raw)
gain, bip = vep_prepare.gain_reorder(bip_gain_inv_minus, bip, bip_name)
roi = vep_prepare.read_vep_mrtrix_lut()
if remove_cerebellar:
    cereb_cortex = ['Left-Cerebellar-cortex', 'Right-Cerebellar-cortex']
    gain.T[roi.index('Left-Cerebellar-cortex')] = np.ones(np.shape(gain.T[-1])) * np.min(gain)
    gain.T[roi.index('Right-Cerebellar-cortex')] = np.ones(np.shape(gain.T[-1])) * np.min(gain)

#%% Plot stim data
onset = 38.25*60
offset = 41.05*60
base_length = 0  # plot ts_on sec before and after
start_idx = int((onset - base_length) * bip.info['sfreq'])
end_idx = int((offset + base_length) * bip.info['sfreq'])

y = bip.get_data()[:, start_idx:end_idx]
t = bip.times[start_idx:end_idx]
# ch_names = bip.ch_names 
ch_names = [ "B1-2", "B2-3", "B4-5", "B5-6", "B6-7",
        "B8-9", "TB1-2",  "TB4-5", "TB6-7",
         "A1-2", "A2-3", "A3-4", "A4-5", "A5-6", "A6-7",
         "A8-9", "A9-10", 
         "Im1-2", "Im3-4", "Im5-6", "Im6-7",
         "Ia1-2","Ia3-4", "Ia5-6", "Ia6-7",]
plot_SEEG_timeseries(t, y, ch_names, bip.ch_names, scaleplt=0.0008, figsize=(20, 20)) # Plot all empirical time series

# %%
onset = 37.6*60
offset = 41.05*60
base_length = 0  # plot ts_on sec before and after
start_idx = int((onset - base_length) * bip.info['sfreq'])
end_idx = int((offset + base_length) * bip.info['sfreq'])

y = bip.get_data()[:, start_idx:end_idx]
t = bip.times[start_idx:end_idx]
# ch_names = bip.ch_names 
# ch_names = [ "B1-2", "B8-9", "TP6-7", "TP7-8",
#          "A1-2", "A2-3", "A3-4", "A5-6", "A9-10", 
#          "Ia3-4", "Ia5-6", "Ia6-7"]
ch_names = ["TB'1-2", "B1-2", "B8-9", "TP1-2", "TP6-7", "TP7-8",
         "A1-2", "A2-3", "A3-4", "A5-6", "A9-10", 
         "Ia3-4", "Ia5-6", "Ia6-7"]
plot_SEEG_timeseries(t, y, ch_names, bip.ch_names, scaleplt=0.0004, figsize=(20, 8)) # Plot all empirical time series

# %%
