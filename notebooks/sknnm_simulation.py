
from src import (
    datahandler as dah,
    model as mod,
    graphs as gra,
    visualize as vis,
    analysis as ana,
)

import pandas as pd
import numpy as np
from brian2 import Hz, mV, ms, Mohm, uF



# Set data paths
path_comp = "../data/manc_completeness.csv"
path_con = "../data/manc_connectivity.parquet"
path_res = "../results/manc_simulations/"

# Load connectome data
ds_ids = dah.load_flywire_ids(path_comp)
df_con = dah.load_flywire_connectivity(path_con, ds_ids)
# Load the table of MANC neurons
neuron_info = pd.read_csv('../data/manc_neurons.csv')

# Create a MANC ID: type and a type: MANC ID dictionary
mancid_type_dict = { key:val for key,val in zip(neuron_info['bodyId'], neuron_info['type'])}
type_mancid_dict = { key:[] for key in neuron_info['type'].unique() }
for mancid, key in mancid_type_dict.items(): 
    type_mancid_dict[key].append(mancid)


# Settings for following simulations
run_exp_kw_args = {
"ds_ids": ds_ids,                   # neuron database IDs
    "df_con": df_con,               # connectivity data
    "path_res": path_res,           # store results here
    "n_proc": -1,                   # number of CPUs to use, -1 uses all available CPU
    "id2name": mancid_type_dict,    # dictionary to map neuron ids to types
    "force_overwrite": True,        # if true, overwrite existing results
    "n_trl": 100                    # number of trials to run
}

# Parameter for the LIF models
params = {
    # network constants
    # Kakaria and de Bivort 2017 https://doi.org/10.3389/fnbeh.2017.00008
    # refereneces therein, e.g. Hodgkin and Huxley 1952
    'v_0'       : -52 * mV,               # resting potential
    'v_rst'     : -52 * mV,               # reset potential after spike
    'v_th'      : -45 * mV,               # threshold for spiking
    't_mbr'     : .002 * uF * 10. * Mohm, # membrane time scale (capacitance * resistance)

    # Jürgensen et al https://doi.org/10.1088/2634-4386/ac3ba6
    'tau'       : 5 * ms,                 # time constant (this is the excitatory one, the inhibitory is 10 ms)

    # Lazar et at https://doi.org/10.7554/eLife.62362
    # they cite Kakaria and de Bivort 2017, but those have used 2 ms
    't_rfc'     : 2.2 * ms,               # refractory period

    # Paul et al 2015 doi: 10.3389/fncel.2015.00029
    't_dly'     : 1.8*ms,                 # delay for changes in post-synaptic neuron

    # empirical 
    'w_syn'     : .275 * mV,              # weight per synapse (note: modulated by exponential decay)    default: 0.275*mV
    'r_poi'     : 150*Hz,                 # default rate of the Poisson input
    'r_poi2'    :  10*Hz,                 # default rate of another Poisson input (useful for different frequencies)
    'f_poi'     : 250,                    # scaling factor for Poisson synapse

    # equations for neurons
    'eqs'       : '''
                    dv/dt = (x - (v - v_0)) / t_mbr : volt (unless refractory)
                    dx/dt = -x / tau                : volt (unless refractory) 
                    rfc                             : second
                    ''',
    # condition for spike
    'eq_th'     : 'v > v_th', 
    # rules for spike        
    'eq_rst'    : 'v = v_rst; w = 0; x = 0 * mV', 
}

# Neurons to activate during experiment
neurons = ['pMP2', 'TN1a']

### Run the experiments
for neuron in neurons:
    # Make instructions
    instructions = [(0, "stim", []), (2, "stim", type_mancid_dict[neuron]), (12, "stim_off", []), (22, "end", [])]

    mod.run_exp(exp_name=f'{neuron}_after_act', exp_inst=instructions, **run_exp_kw_args, params=params)

# Experiment duration
duration = instructions[-1][0]   # [s]

# Create paths from which to load results
outputs = [ f'../results/manc_simulations/{neuron}_after_act.parquet' for neuron in neurons ]
# Load spike times
df_spkt = ana.load_exps(outputs)
# List of neuron sto analyse
ana_neurons = ['ps1 MN', 'i2 MN', 'hg1 MN', 'hg2 MN', 'hg3 MN', 'hg4 MN', 'tp1 MN', 'tp2 MN', 'tpn MN', 'TTMn', 'b1 MN', 'b2 MN', 'b3 MN', 'DLMn a, b', 'DLMn c-f', 'DVMn 3a, b', 'hDVM MN', 'MNwm35']

# Transform list of types into list of IDs
ana_neu = []
for neuron in ana_neurons:
    ana_neu.extend(type_mancid_dict[neuron])

# Get continuous spike rates for all neurons of interest
hist_edges, cont_spike_rate, exp_names = ana.get_rate_continuous(df_spkt, ana_neu, bin_width=0.01, t0=0, tend=duration)
neuron_ids = []
spk_rates = []
# Save rates and IDs into lists
for key, val in cont_spike_rate.items():
    neuron_ids.append(key)
    spk_rates.append(val)

# Stack spike rates
spk_rates = np.stack(spk_rates, axis=0)
# Export spike rates
np.savez('../results/manc_simulations/spk_rates.npz', spike_rates=spk_rates, neuron_ids=neuron_ids, exp_names=exp_names)