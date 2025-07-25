
import pathlib
from pathlib import Path

import pandas as pd
import numpy as np

    
def load_exps(l_prq, load_pickle=True):
    '''Load simulation results from disk

    Parameters
    ----------
    l_prq : list
        List of pickle files with simulation results
    load_pickle : bool, optional
        If True, load some metadata from pickle file, default True

    Returns
    -------
    exps : df
        data for all experiments 'path_res'
    '''
    # cycle through all experiments
    dfs, stim_ids = [], dict()
    for p in l_prq:

        # ensure path object
        p = Path(p)

        # load spike data from parquet file
        df = pd.read_parquet(p)
        df.loc[:, 't'] = df.loc[:, 't'].astype(float)
        dfs.append(df)

        if load_pickle: # load pickle for metadata
            
            # workaround: legacy pickle files may have OS-dependent
            # path objects, newer ones have strings
            try:
                posix_backup = pathlib.PosixPath
                pkl = pd.read_pickle(p.with_suffix('.pickle'))
            except NotImplementedError:
                try:
                    # currently on linux, created on windows
                    pathlib.WindowsPath = pathlib.PosixPath
                    pkl = pd.read_pickle(p.with_suffix('.pickle'))
                except NotImplementedError:
                    # currently on windows, created on linux
                    pathlib.PosixPath = pathlib.WindowsPath
                    pkl = pd.read_pickle(p.with_suffix('.pickle'))
            finally:
                pathlib.PosixPath = posix_backup


            # get stimulated neurons
            df_inst = pkl['df_inst']
            rows = df_inst.loc[:, 'mode'].str.startswith('stim')
            ids = [ i for j in df_inst.loc[rows].loc[:, 'id'] for i in j]
            stim_ids[pkl['exp_name']] = ids


    df = pd.concat(dfs)

    # TODO: attrs is experimental, find another way https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.attrs.html#pandas.DataFrame.attrs
    df.attrs['stim_ids'] = stim_ids

    return df

def get_rate(df, duration, n_trl=30):
    '''Calculate rate and standard deviation for all experiments
    in df

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe generated with `load_exps` containing spike times
    duration : float
        Trial duration in seconds
    n_trl : int, optional
        Number of trials, by default 30

    Returns
    -------
    df_rate : pd.DataFrame
        Dataframe with average firing rates
    df_std : pd.DataFrame
        Dataframe with standard deviation of firing rates
    '''

    rate, std, id, exp_name = [], [], [], []

    for e, df_e in df.groupby('exp_name'):
        for i, df_i in df_e.groupby('database_id'):

            r = np.zeros(n_trl)
            for t, df_t in df_i.groupby('trial'):
                r[t] = len(df_t) / duration

            rate.append(r.mean())
            std.append(r.std())
            id.append(i)
            exp_name.append(e)

    d = {
        'r' : rate,
        'std': std,
        'database_id' : id,
        'exp_name' : exp_name,
    }
    df = pd.DataFrame(d)
    
    df_rate = df.pivot_table(columns='exp_name', index='database_id', values='r')
    df_std = df.pivot_table(columns='exp_name', index='database_id', values='std')

    df_rate = df_rate.fillna(0)
    df_std = df_std.fillna(0)

    return df_rate, df_std



def rename_index(df, id2name):
    '''Rename database IDs to custom neuron names in index
    Also sort index and columns

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with database IDs as index
    name2id : dict
        Mapping between custom neuron names and database IDs

    Returns
    -------
    df : pd.DataFrame
        Renamed and sorted dataframe
    '''

    # replace database IDs with custom names
    df = df.rename(index=id2name)

    # sort: str first (i.e. custom names), then int (i.e. database IDs)
    df.index = df.index.astype(str)
    df = df.loc[
        sorted(sorted(df.index.astype(str)), key=lambda x: (x[0].isdigit(), x)), 
        sorted(df.columns.sort_values(), key=lambda x: len(x.split('+')))
        ]
    
    return df



def save_xls(df, path):
    '''Save DataFrame as xls file

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with experiments as columns and neurons as indixes
    path : str
        Filename of the xls file
    '''

    print('INFO: saving {} experiments to {}'.format(len(df.columns), path))

    with pd.ExcelWriter(path, mode='w', engine='xlsxwriter') as w:

        # write to file
        df.to_excel(w, sheet_name='all_experiments')

        # formatting in the xlsx file
        wb = w.book

        # set floating point display precision here (excel format)
        fmt = wb.add_format({'num_format': '#,##0.0'}) 
        for _, ws in w.sheets.items():
            ws.set_column(1, 1, 10, fmt)
            ws.freeze_panes(1, 1)



def get_rate_continuous(df_spkt, neu, bin_width=20, t0=0, tend=1):
    '''
    
    '''
    # Make bin array
    bins = np.arange(t0, tend, bin_width)

    spk_rates = {}
    exp_names = []
    
    for i, (e, df_exp) in enumerate(df_spkt.groupby('exp_name')):
        # Group by database ID
        gr_neu = df_exp.groupby('database_id')

        # Iterate over the requested neurons
        for j, n in enumerate(neu):
            idx = int(n)
            try:
                # Get data for current neuron
                df_neu = gr_neu.get_group(idx)
                # 
                spike_mat = np.hstack([ np.sort(df_trl['t']) for _, df_trl in df_neu.groupby('trial') ])
                
                if spike_mat.shape[0] > 0:
                    # Calculate histogram
                    hist, hist_edges = np.histogram(spike_mat, bins, density=False)
                    # Convert to spikes per second
                    hist = hist / bin_width / (df_spkt['trial'].max() + 1)
                else:
                    hist = np.zeros(bins.shape[0]-1)

            except KeyError:
                hist = np.zeros(bins.shape[0]-1)
                pass

            if i > 0:
                    spk_rates[n] = np.vstack([spk_rates[n], hist])
            else:
                spk_rates[n] = hist
        
        exp_names.append(e)
    
    return hist_edges, spk_rates, exp_names