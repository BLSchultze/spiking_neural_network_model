
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore

import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
import seaborn as sns
sns.set_style('whitegrid')
plt.rcParams['savefig.facecolor'] = 'w'



def plot_raster(df_spkt, neu, id2name=dict(), xlims=(None, None), figsize=(), path=None):
    '''Plot raster plots for given experiments and neurons

    Parameters
    ----------
    df_spkt : pd.DataFrame
        Each row contains a spike event
    neu : list
        List of database IDs as appearing in df_spkt.
        `neu` can also contain custom neuron names, but in this case `name2id`
        must be supplied
    id2name : dict, optional
        Mapping between database IDs and neuron types, by default dict()
    xlims : tuple, optional
        xlims for plot, by default (None, None)
    figsize : tuple, optional
        dimension of the plot, passed to plt.subpolots
    path : str, optional
        Filename for saving the plot, by default None
    '''

    exp = df_spkt.loc[:, 'exp_name'].unique()
    n_exp, n_neu = len(exp), len(neu)

    if figsize:
        dx, dy = figsize
    else:
        dx, dy = 3*n_neu, 2*n_exp
    print('INFO: setting figsize to ({}, {})'.format(dx, dy))

    fig, axmat = plt.subplots(ncols=n_neu, nrows=n_exp, squeeze=False, figsize=(dx, dy))

    # Add names to the data frame
    df_spkt['name'] = df_spkt['database_id'].map(lambda l: id2name.get(l, l))

    for i, (e, df_exp) in enumerate(df_spkt.groupby('exp_name')):

        trl_max = df_exp.max()['trial'] # for axis limits

        gr_neu = df_exp.groupby('database_id')
        for j, n in enumerate(neu):
            ax = axmat[i,j]

            idx = int(n)

            try:
                df_neu = gr_neu.get_group(idx)
            
                for trl, df_trl in df_neu.groupby('trial'):
                    t = df_trl.loc[:, 't']
                    ax.eventplot(t, lineoffset=trl, linewidths=.5)

            except KeyError:
                pass
            
            # formatting
            if j == 0:
                ax.set_ylabel(e)
            else:
                ax.set_yticklabels('')
                
            if i == 0:
                ax.set_title(id2name.get(n, n))

            ax.grid(None)
            ax.set_xlim(xlims)
            ax.set_ylim(-0.5, trl_max + 0.5)
           

    for ax in axmat[-1]:
        ax.set_xlabel('time [s]')
    fig.tight_layout()

    if path:
        fig.savefig(path)



def plot_rate(df_spkt, neu, xlims, sigma=25, n_trl=30, do_zscore=False, id2name=dict(), figsize=(), path=None):
    '''Plot rates for given experiments and neurons

    Parameters
    ----------
    df_spkt : pd.DataFrame
        Each row contains a spike event
    neu : list
        List of database IDs as appearing in df_spkt.
        `neu` can also contain custom neuron names, but in this case `name2id`
        must be supplied
    xlims : tuple
        xlims for plot [s]
    sigma : float, optional
        standard deviation for Gaussian kernel for smoothing [ms], by default 25
    n_trl : int, optional
        number of trials to calculate the avg rate, by default 30
    do_score : bool, optional
        If True, zscore the firing rate for each neuron, by default False
    id2name : dict, optional
        Mapping between database IDs and neuron types, by default dict()
    figsize : tuple, optional
        dimension of the plot, passed to plt.subplots
    path : str, optional
        Filename for saving the plot, by default None
    '''

    exp = df_spkt.loc[:, 'exp_name'].unique()
    n_exp = len(exp)

    if figsize:
        dx, dy = figsize
    else:
        dx, dy = 5 * n_exp, 4
    print('INFO: setting figsize to ({}, {})'.format(dx, dy))

    fig, axarr = plt.subplots(ncols=n_exp, squeeze=False, figsize=(dx, dy))

    # Add names to the data frame
    df_spkt['name'] = df_spkt['database_id'].map(lambda l: id2name.get(l, l))

    gr_exp = df_spkt.groupby('exp_name')
    
    bins = np.arange(*xlims, 1e-3)

    for i, e in enumerate(exp):
        ax = axarr[0][i]

        df_exp = gr_exp.get_group(e)
        gr_neu = df_exp.groupby('database_id')

        df_bin = pd.DataFrame()

        for n in neu:
            idx = int(n)

            try:
                df_neu = gr_neu.get_group(idx)
                gr_trl = df_neu.groupby('trial')

                for trl in range(n_trl):

                    try: 
                        df_trl = gr_trl.get_group(trl)
                        t = df_trl.loc[:, 't']
                    except KeyError:
                        t = []

                    y, _ = np.histogram(t, bins=bins)
                    y = gaussian_filter1d(y.astype(float), sigma=sigma, axis=0)
                    y *= 1e3
                    df = pd.DataFrame(data={
                        't' : bins[:-1],
                        'r': y,
                        'trl': trl,
                        'neu': id2name.get(n, n),
                    })
                    df_bin = pd.concat([df_bin, df], ignore_index=True)

            except KeyError:
                df = pd.DataFrame(data={
                    't' : bins[:-1],
                    'r': 0,
                    'neu': id2name.get(n, n),
                })
                df_bin = pd.concat([df_bin, df], ignore_index=True)

        if do_zscore:
            for n, df in df_bin.groupby('neu'):
                idx = df.index
                df_bin.loc[idx, 'r'] = zscore(df_bin.loc[idx, 'r'], ddof=1)

        sns.lineplot(data=df_bin, ax=ax, x='t', y='r', errorbar='sd', hue='neu')

        # formatting
        ax.legend()
        ax.set_title(e)
        ax.set_xlabel('time [s]')
        ax.set_ylabel('rate [Hz]')

    fig.tight_layout()
    if path:
        fig.savefig(path)



def plot_rate_heatmap(df_spkt, neu, xlims, sigma=25, n_trl=30, do_zscore=False, exclude_stim=False, color_range=(None, None), id2name=dict(), figsize=(), path=None):
    '''Plot rates for given experiments and neurons in a heatmap

    Parameters
    ----------
    df_spkt : pd.DataFrame
        Each row contains a spike event
    neu : list
        List of database IDs as appearing in df_spkt.
        `neu` can also contain custom neuron names, but in this case `name2id`
        must be supplied
    xlims : tuple
        xlims for plot [s]
    sigma : float, optional
        standard deviation for Gaussian kernel for smoothing [ms], by default 25
    n_trl : int, optional
        number of trials to calculate the avg rate, by default 30
    do_score : bool, optional
        If True, zscore the firing rate for each neuron, by default False
    exclude_stim : bool, optional
        If True, replace stimulated neurons with nan, by default False
    color_range : tuple, optional
        Values for min and max for the color map, by default (None, None)
    id2name : dict, optional
        Mapping between database IDs and neuron types, by default dict()
    figsize : tuple, optional
        dimension of the plot, passed to plt.subplots
    path : str, optional
        Filename for saving the plot, by default None
    '''

    exp = df_spkt.loc[:, 'exp_name'].unique()
    n_exp, n_neu = len(exp), len(neu)
    if figsize:
        dx, dy = figsize
    else:
        dx, dy = 5 * n_exp, .25 * n_neu + 1
    print('INFO: setting figsize to ({}, {})'.format(dx, dy))

    fig, axarr = plt.subplots(ncols=n_exp, squeeze=False, figsize=(dx, dy))

    # Add names to the data frame
    df_spkt['name'] = df_spkt['database_id'].map(lambda l: id2name.get(l, l))
    gr_exp = df_spkt.groupby('exp_name')
    
    bins = np.arange(*xlims, 1e-3)

    if do_zscore:
        cmap = 'coolwarm'
        norm = CenteredNorm()
    else: 
        cmap = 'viridis'
        norm = None

    for i, e in enumerate(exp):
        ax = axarr[0][i]

        df_exp = gr_exp.get_group(e)
        gr_neu = df_exp.groupby('database_id')

        # stuff for excluding stim
        # TODO: make more pretty
        id_b = df_spkt.attrs['stim_ids'][e]
        b2f = pd.Series(df_exp.loc[:, 'database_id'].values, index=df_exp.loc[:, 'brian_id']).to_dict()
        id_f = [ b2f[i] for i in id_b ]

        Z = []
        for n in neu:
            idx = int(n)

            try:
                df_neu = gr_neu.get_group(idx)
                t = df_neu.loc[:, 't']
            except KeyError:
                t = []

            z, _ = np.histogram(t, bins=bins)
            z = gaussian_filter1d(z.astype(float), sigma=sigma, axis=0)
            z = z / n_trl * 1e3
            if do_zscore:
                z = zscore(z, ddof=1)

            if exclude_stim and idx in id_f:
                z[:] = np.nan

            Z.append(z)

        Z = np.vstack(Z)
        x = bins[:-1]
        y = np.arange(n_neu)
        im = ax.pcolormesh(x, y, Z, cmap=cmap,  norm=norm, vmin=color_range[0], vmax=color_range[1])
        fig.colorbar(im, ax=ax, location='right', orientation='vertical')

        # TODO colorbar label and xlabel

        ax.set_yticks(y)
        ax.set_yticklabels([ id2name.get(n, n) for n in neu ])

        # formatting
        ax.set_title(e)
        ax.set_xlabel('time [s]')

    fig.tight_layout()

    if path:
        fig.savefig(path)



def firing_rate_matrix(df_rate, rate_change=False, scaling=.5, path='', id2name={}):
    '''Plot heatmap showing the firing rates of neurons in different experiments
 
    Parameters
    ----------
    df_rate : pd.DataFrame
        Rate data with experiments as columns and neurons as index
    rate_change : bool, optional
        If True, use diverging colormap and center on 0, by default False
    scaling : float, optional
        Scales figure size, by default .5
    path : path-like, optional
        Filename for saving the plot, by default ''
    id2name: dict
        Mapping database IDs to neuron types, default {}
    '''
    # Sort spike rates
    df_rate.sort_values(by=df_rate.columns.to_list(), ascending=False, inplace=True)
    # Rename index using id2name
    df_rate.rename(id2name, inplace=True)

    # figure dimensions
    n_neu, n_exp = df_rate.shape
    figsize = (scaling*n_exp, scaling*n_neu)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title('firing rate [Hz]')

    
    if rate_change: # plot settings for rate changes
        heatmap_kw_args = {
            'cmap': 'coolwarm',
            'center': 0,
        }
    else: # plot settings for absolute rates
        heatmap_kw_args = {
            'cmap': 'viridis',
        }

    sns.heatmap(
        ax=ax, data=df_rate, square=True,
        xticklabels=True, yticklabels=True,
        annot=True, fmt='.1f', annot_kws={'size': 'small'},
        cbar=False, **heatmap_kw_args,
    )
    ax.tick_params(axis='x', labeltop=True, labelbottom=True, labelrotation=90)
    
    if path:
        fig.savefig(path)
        plt.close(fig)



def plot_isi_hist(df_spkt, neu, id2name=dict(), xlims=None, figsize=(), path=None, bins=20):
    '''Plot inter-spike-intervals for given experiments and neurons

    Parameters
    ----------
    df_spkt : pd.DataFrame
        Each row contains a spike event
    neu : list
        List of database IDs as appearing in df_spkt.
        `neu` can also contain custom neuron names, but in this case `name2id`
        must be supplied
    id2name : dict, optional
        Mapping between database IDs and neuron types, by default dict()
    xlims : tuple, optional
        xlims for plot, by default None
    figsize : tuple, optional
        dimension of the plot, passed to plt.subpolots
    path : str, optional
        Filename for saving the plot, by default None
    bins : int | array | list
        bin count for the histogram or list/array with bin edges
    '''

    exp = df_spkt.loc[:, 'exp_name'].unique()
    n_exp, n_neu = len(exp), len(neu)

    if figsize:
        dx, dy = figsize
    else:
        dx, dy = 3*n_neu, 2*n_exp
    print('INFO: setting figsize to ({}, {})'.format(dx, dy))

    fig, axmat = plt.subplots(ncols=n_neu, nrows=n_exp, squeeze=False, figsize=(dx, dy))

    # Add names to the data frame
    df_spkt['name'] = df_spkt['database_id'].map(lambda l: id2name.get(l, l))

    for i, (e, df_exp) in enumerate(df_spkt.groupby('exp_name')):
        # Group by database ID
        gr_neu = df_exp.groupby('database_id')

        # Iterate over the requested neurons
        for j, n in enumerate(neu):
            ax = axmat[i,j]
            idx = int(n)
            try:
                # Get data for current neuron
                df_neu = gr_neu.get_group(idx)
                # Calculate inter-spike-intervals for all trials
                spike_intervals = np.hstack([ np.diff(np.sort(df_trl['t'])) for _, df_trl in df_neu.groupby('trial') ]) * 1000     # convert to ms
                
                if spike_intervals.shape[0] > 0:
                    # Calculate histogram
                    hist, hist_edges = np.histogram(spike_intervals, bins, density=True) 
                    # Plot histogram as curve
                    ax.plot(hist_edges[:-1], hist)

            except KeyError:
                pass
            
            # formatting
            if j == 0:
                ax.set_ylabel(e)
            else:
                ax.set_yticklabels('')
                
            if i == 0:
                ax.set_title(id2name.get(n, n))

            ax.grid(None)
            if xlims != None:
                ax.set_xlim(xlims)           

    for ax in axmat[-1]:
        ax.set_xlabel('inter-spike-interval [ms]')
    fig.tight_layout()

    if path:
        fig.savefig(path)



def plot_rates_single(df_spkt, neu, id2name=dict(), xlims=None, figsize=(), path=None, bin_width=20, t0=0, tend=1):
    '''Plot inter-spike-intervals for given experiments and neurons

    Parameters
    ----------
    df_spkt : pd.DataFrame
        Each row contains a spike event
    neu : list
        List of database IDs as appearing in df_spkt.
        `neu` can also contain custom neuron names, but in this case `name2id`
        must be supplied
    id2name : dict, optional
        Mapping between database IDs and neuron types, by default dict()
    xlims : tuple, optional
        xlims for plot, by default None
    figsize : tuple, optional
        dimension of the plot, passed to plt.subpolots
    path : str, optional
        Filename for saving the plot, by default None
    bins : int | array | list
        bin count for the histogram or list/array with bin edges
    '''

    # Make bin array
    bins = np.arange(t0, tend, bin_width)

    # Get number of experiments and neurons to plot
    exp = df_spkt.loc[:, 'exp_name'].unique()
    n_exp, n_neu = len(exp), len(neu)

    if not figsize:
        figsize = (3*n_neu, 2*n_exp)

    # New figure
    fig, ax = plt.subplots(n_exp, n_neu, figsize=figsize)
    
    # Iterate over the experiments
    for i, (e, df_exp) in enumerate(df_spkt.groupby('exp_name')):
        # Group by database ID
        gr_neu = df_exp.groupby('database_id')

        # Iterate over the requested neurons
        for j, n in enumerate(neu):
            idx = int(n)
            try:
                # Get data for current neuron
                df_neu = gr_neu.get_group(idx)
                # Collect all spikes over all trials in one array
                spike_train = np.hstack([ np.sort(df_trl['t']) for _, df_trl in df_neu.groupby('trial') ])
                
                if spike_train.shape[0] > 0:
                    # Calculate histogram
                    hist, hist_edges = np.histogram(spike_train, bins, density=False)
                    # Convert to spikes per second
                    hist = hist / bin_width / (df_spkt['trial'].max() + 1)
                else:
                    hist = np.zeros(bins.shape[0]-1)

            except KeyError:
                hist = np.zeros(bins.shape[0]-1)
                pass
            
            # Plot spike rate (PSTH)
            ax[i,j].plot(hist_edges[:-1], hist)
    
            # Add title in first row
            if i == 0:
                ax[i,j].set_title(id2name.get(n, n))
            # Y-label in first column
            if j == 0:
                ax[i,j].set_ylabel(e + '\nspike rate [Hz]')
            # Adjust x limits if requested
            if xlims != None:
                ax[i,j].set_xlim(xlims)
                    
    # X-label in last row
    for cax in ax[-1]:
        cax.set_xlabel('time [s]')
    fig.tight_layout()

    if path:
        fig.savefig(path)