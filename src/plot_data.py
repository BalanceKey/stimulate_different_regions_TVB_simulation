'''
Plot data
'''
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
from collections import OrderedDict
import glob


def plot_pattern(pattern_object, roi):
    """
    pyplot in 2D the given X, over T.
    """
    plt.figure(figsize=(15, 10), tight_layout=True)
    plt.subplot(211)
    plt.bar(np.r_[0:pattern_object.spatial_pattern.shape[0]], pattern_object.spatial_pattern[:, 0])
    plt.xticks(np.r_[:len(roi)], roi, rotation=90, size=6)
    plt.title("Space")
    plt.subplot(212)
    plt.plot(pattern_object.time.T, pattern_object.temporal_pattern.T, linewidth=0.5)
    plt.title("Time")
    plt.show()


def plot_SEEG_timeseries(t, y, ch_names, seeg_info=None, scaleplt=0.002):
    '''Plots all SEEG timeseries'''
    fig = plt.figure(figsize=(40, 80))

    for ind, ich in enumerate(ch_names):
        plt.plot(t, scaleplt * (y[ind, :]) + ind, 'blue', lw=0.5);

    if seeg_info is not None:
        vlines = [seeg_info['onset'], seeg_info['offset']]
        for x in vlines:
            plt.axvline(x, color='DeepPink', lw=3)

    plt.xticks(fontsize=26)
    plt.ylim([-1, len(ch_names) + 0.5])
    plt.xlim([t[0], t[-1]])
    plt.yticks(np.arange(len(ch_names)), ch_names, fontsize=26)
    plt.gcf().subplots_adjust(left=0.2)
    plt.gcf().subplots_adjust(top=0.97)
    plt.xlabel('Time', fontsize=50)
    plt.ylabel('Electrodes', fontsize=50)
    plt.title('SEEG recording', fontweight='bold', fontsize=50)
    plt.tight_layout()
    plt.show()

def get_all_seizures(pid, types=None, verbose=True, raw_dir='/data/epinov', key_string='seizure'):
    subj_raw_dir = op.join(raw_dir, pid)
    vhdr_pattern = op.join(subj_raw_dir, f'ses-01/ieeg/*{key_string}*.vhdr')
    all_seizures = glob.glob(vhdr_pattern)
    if types is None:
        seizures = OrderedDict([(s, seizure) for s, seizure in enumerate(all_seizures)])
    else:
        seizures = OrderedDict([(s, seizure) for s, seizure in enumerate(all_seizures) if np.any([t in op.basename(seizure) for t in types])])
    if verbose:
        if types is None:
            print('All seizures:')
        else:
            print("All seizures of type [" + ", ".join(["'" + type + "'" for type in types]) + "]:")
        for s in seizures.items():
            print(f'[{s[0]}] {s[1]}')
    return list(seizures.values())

def plot_sources_and_sensors_3D(sources, sensors):
    """
    Plots in 3D the sensors and sources
    :return: a 3D plot interactive for visualisation of brain regions and electrodes
    """
    # TODO