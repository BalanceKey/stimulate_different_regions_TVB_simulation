from tvb.simulator.lab import *
import numpy as np
import time as tm
import matplotlib.pyplot as plt

def run_simulation(stimulation_parameters, init_conditions, dt, epileptors, con, coupl, heunint, mon_tavg, bip_names, bip_gain_prior_norm, roi, plot=False):
    print(f'Stimulating with parameters: {stimulation_parameters}')

    sfreq = stimulation_parameters['sfreq']/4                         # how many steps there are in one second
    onset = 2 * sfreq                                                 # stimulation onset
    stim_length = stimulation_parameters['duration'] * sfreq + onset  # stimulation duration
    T = 1 / stimulation_parameters['freq'] * sfreq                    # pulse repetition period [s]
    tau = stimulation_parameters['tau']/1000000 * sfreq               # pulse duration, number of steps [microsec]
    I = stimulation_parameters['amp']                                 # pulse intensity [mA]

    simulation_length = stim_length + 4 * sfreq
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

    # Stimulus weights
    choi = stimulation_parameters['choi']
    # Estimated stimulation weights using stimulation location + gain matrix
    try:
        idx = bip_names.index(choi)
    except:
        print(f"{choi} not in bip_names list ! Resuming simulation... ")
        exit()

    if plot:
        fig = plt.figure(figsize=(25, 10), tight_layout=True)
        # img = plt.bar(np.r_[0:bip_gain_prior[idx].shape[0]], bip_gain_prior[idx], color='blue', alpha=0.5)
        img = plt.bar(np.r_[0:bip_gain_prior_norm[idx].shape[0]], bip_gain_prior_norm[idx], color='red', alpha=0.5)
        plt.xticks(np.r_[:len(roi)], roi, rotation=90)
        plt.ylabel('Gain matrix for channel ' + choi, fontsize=30)
        plt.show()

    E_magnitude = bip_gain_prior_norm[idx]  # normalized gain matrix prior

    stim_weights = E_magnitude 
    stimulus = patterns.StimuliRegion(temporal=eqn_t,
                                    connectivity=con,
                                    weight=stim_weights)
    stimulus.configure_space()
    stimulus.configure_time(np.arange(0., np.size(stimulus_ts), 1))
    if plot:
        plot_pattern(stimulus, roi)

    #%% Initial conditions
    # ic = [-1.4, -9.6, 2.97, 0.0]
    #%% Simulator
    sim = simulator.Simulator(model=epileptors,
                            initial_conditions=init_conditions,
                            connectivity=con,
                            stimulus=stimulus,
                            coupling=coupl,
                            integrator=heunint,
                            conduction_speed=np.inf,
                            monitors=[mon_tavg])
    sim.configure()

    print("Starting simulation...")
    tic = tm.time()
    ttavg = sim.run(simulation_length=simulation_length)
    print('Simulation ran for ', (tm.time() - tic) / 60.0, 'mins')
    return ttavg