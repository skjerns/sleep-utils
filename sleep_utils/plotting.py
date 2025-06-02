# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 10:50:29 2018

@author: SimonKern


This contains all functions related to plotting,
Eg. plotting
    - hypnograms,
    - spectograms
    - etc etc etc
"""
import os
import matplotlib
import time
import scipy
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from lspopt import spectrogram_lspopt
from PIL import Image
import warnings
import seaborn as sns
import pandas as pd
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import zscore

# imports for backwards compatibility
from sleep_utils.gui import choose_file, display_textbox, display_listbox


def plotmulti(*args):
    """
    plots multiple signals, each signal in a new subplot.
    """
    plt.figure()
    for i, sig in enumerate(args):
        plt.subplot(len(args), 1, i+1)
        plt.plot(sig, linewidth=0.5)


def color_background(cvalues, stepsize=None, cmap='RdYlGn_r'):
    """
    Shade the background of the current axis using some x-positions

    This will need a pair of xlimits (2-tuple)
    and a value. The value can either be an RGB value or a
    float 0-1 with a given cmap.

    :param cvalues: the intensity that it should be coloured
    :param stepsize: the length of the interval for each value
                     if None, will be set to xlim/len(cvalues)
    :param cmap: a color map, standard a green-red will be used
    """
    # if stepsize is unknown, we infer space the cvalues evenly across the plot
    if isinstance(cmap, str):
        cmap = matplotlib.cm.get_cmap(cmap)

    if stepsize is None:
        rxlim = plt.xlim()[1]
        rmargin = plt.margins()[0]
        stepsize = int((rxlim-(rxlim*rmargin))/len(cvalues)) + 1

    for i, v in enumerate(cvalues):
        i1, i2 = i*stepsize, (i+1)*stepsize
        plt.axvspan(i1, i2, facecolor=v2rgb(v, cmap), alpha=0.4)
#        plt.axvline(i*stepsize, linewidth=0.5, c= 'black', alpha=0.1)
    return None


def v2rgb(v, cmap=None):
    import matplotlib.colors as mcolors
    if cmap is None:
        cdict = {'red':   ((0.0, 0.0, 0.0),
                           (0.5, 0.0, 0.0),
                           (1.0, 1.0, 1.0)),
                 'blue':  ((0.0, 0.0, 0.0),
                           (1.0, 0.0, 0.0)),
                 'green': ((0.0, 0.0, 1.0),
                           (0.5, 0.0, 0.0),
                           (1.0, 0.0, 0.0))}

        cmap = mcolors.LinearSegmentedColormap(
            'my_colormap', cdict, 100)
    return cmap(v)


def plot_confusion_matrix(confmat, target_names=None, ax=None, title='',
                          cmap='Blues', perc=True, cbar=True,
                          xlabel='True sleep stage', ylabel='Predicted sleep stage'):
    """
    Plot a Confusion Matrix.

    :param conf_mat: A confusion matrix (2D square matrix)
    :param target_names: Names of the classes
    :param title: The title of the figure
    :param cmap: The color map (matplotlib cmap)
    :param perc: Show values as percentages
    :param
    :param cbar: display a color bar or not
    """
    if ax is None:
        plt.figure()
        ax = plt.subplot(1, 1, 1)

    c_names = []
    r_names = []
    if target_names is None:
        target_names = [str(i) for i in np.arange(len(confmat))]

    for i, label in enumerate(target_names[:max(confmat.shape)]):
        c_names.append(label + '\n(' + str(int(np.sum(confmat[:, i]))) + ')')
        align = max(1, len(str(int(np.sum(confmat[i, :])))) + 3 - len(label))
        r_names.append('{:{align}}'.format(label, align=align) +
                       '\n(' + str(int(np.sum(confmat[i, :]))) + ')')

    cm = confmat
    div = (cm.sum(axis=1)[:, np.newaxis])
    div[div == 0] = 1
    cm = 100 * cm.astype('float') / div

    df = pd.DataFrame(data=np.sqrt(cm), columns=c_names, index=r_names)
    g = sns.heatmap(df, annot=cm if perc else confmat, fmt=".1f" if perc else ".2f",
                    linewidths=.5, vmin=0, vmax=np.sqrt(100), cmap=cmap,
                    cbar=cbar, ax=ax)
    g.set_title(title)
    if cbar:
        cbar = g.collections[0].colorbar
        cbar.set_ticks(np.sqrt(np.arange(0, 100, 20)))
        cbar.set_ticklabels(np.arange(0, 100, 20))
    g.set_ylabel(xlabel)
    g.set_xlabel(ylabel)

    plt.pause(0.1)
    plt.tight_layout()


def plot_noise_std(raw, picks=None, epoch_len=10, ax=None):
    """plot the standard deviation of the channels"""

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=[14, 8])

    ax.clear()

    data = raw.get_data(picks=picks)
    sfreq = raw.info['sfreq']

    data = data[:, :int(len(raw)//(30*sfreq)*epoch_len*sfreq)]
    data = data.reshape([len(data), -1, int(epoch_len*sfreq)])
    stds = np.std(data, axis=-1) # get std per epoch as noise marker

    # this is rather stupid, but worth a try, 5 times median std
    vmax = np.median(np.median(stds, 1))*5
    # flatline detection at 100th of the medians
    vmin_threshold = vmax/100/5
    vmin = (vmax-vmin_threshold)/256 + vmin_threshold

    notnormal = scipy.stats.normaltest(data, axis=-1)[0] > 4000

    stds[notnormal] = vmin

    cmap = cm.get_cmap('RdYlGn_r', 256)
    newcolors = cmap(np.linspace(0, 1, 256))
    gray = np.array([0.7, 0.7, 0.7, 1])
    newcolors[:1, :] = gray
    newcmp = ListedColormap(newcolors)

    ch_names = raw.ch_names if picks is None else picks
    im = ax.imshow(stds, cmap=newcmp, aspect='auto', interpolation='None', vmin=vmin, vmax=vmax)
    ax.set_yticks(np.arange(len(data)), ch_names, fontsize=6)
    # ax.set_xticks(np.arange(len(data)), raw_orig.ch_names, fontsize=6)
    formatter = FuncFormatter(lambda x, pos: '{:.0f}'.format(x * epoch_len))
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlabel('Seconds')
    ax.set_ylabel('Channels')
    ax.set_title(f'Standard deviation of all channels {os.path.basename(raw._filenames[0])} for {epoch_len} second segments')
    return stds, im

def plot_hypnogram(stages, labeldict=None, title=None, epochlen=30, ax=None,
                   starttime=None, verbose=True, xlabel=True, ylabel=True,
                   **kwargs,):
    """
    Plot a hypnogram, the flexible way.

    A labeldict should give a mapping which integer belongs to which class
    E.g labeldict = {0: 'Wake', 4:'REM', 1:'S1', 2:'S2', 3:'SWS'}
    or {0:'Wake', 1:'Sleep', 2:'Sleep', 3:'Sleep', 4:'Sleep', 5:'Artefact'}

    The order of the labels on the plot will be determined by the order of the dictionary.

    E.g.  {0:'Wake', 1:'REM', 2:'NREM'}  will plot Wake on top, then REM, then NREM
    while {0:'Wake', 2:'NREM', 1:'NREM'} will plot Wake on top, then NREM, then REM

    This dictionary can be infered automatically from the numbers that are present
    in the hypnogram but this functionality does not cover all cases.

    :param stages: An array with different stages annotated as integers
    :param labeldict: An enumeration of labels that correspond to the integers of stages
    :param title: Title of the window
    :param epochlen: How many seconds is one epoch in this annotation
    :param ax: the axis in which we plot
    :param verbose: Print stuff or not.
    :param xlabel: Display xlabel ('Time after record start')
    :param ylabel: Display ylabel ('Sleep Stage')
    :param kwargs: additional arguments passed to plt.plot(), e.g. c='red'
    """

    if labeldict is None:
        if np.max(stages) == 1 and np.min(stages) == 0:
            labeldict = {0: 'W', 1: 'S'}
        elif np.max(stages) == 2 and np.min(stages) == 0:
            labeldict = {0: 'W', 2: 'REM', 1: 'NREM'}
        elif np.max(stages) == 4 and np.min(stages) == 0:
            if 1 in stages:
                labeldict = {0: 'W', 4: 'REM', 1: 'S1', 2: 'S2', 3: 'SWS', }
            else:
                labeldict = {0: 'W', 4: 'REM', 2: 'S2', 3: 'SWS'}
        elif np.max(stages) == 9 and np.min(stages) == 0:
            if 1 in stages:
                labeldict = {0: 'W', 4: 'REM',
                             1: 'S1', 2: 'S2', 3: 'SWS', 9: 'A'}
            else:
                labeldict = {0: 'W', 4: 'REM', 2: 'S2', 3: 'SWS'}
        else:
            if verbose:
                print('could not detect labels?')
            if 1 in stages:
                labeldict = {0: 'W', 4: 'REM',
                             1: 'S1', 2: 'S2', 3: 'SWS', 5: 'A'}
            else:
                labeldict = {0: 'W', 4: 'REM', 2: 'S2', 3: 'SWS', 5: 'A'}
        if -1 in stages:
            labeldict['ARTEFACT'] = -1
        if verbose:
            print('Assuming {}'.format(labeldict))

    # check if all stages that are in the hypnogram have a corresponding label in the dict
    for stage in np.unique(stages):
        if not stage in labeldict:
            print(
                'WARNING: {} is in stages, but not in labeldict, stage will be ??'.format(stage))

    # create the label order
    labels = [labeldict[l] for l in labeldict]
    labels = sorted(set(labels), key=labels.index)

    # we iterate through the stages and fetch the label for this stage
    # then we append the position on the plot of this stage via the labels-dict
    x = []
    y = []
    rem_start = []
    rem_end = []
    for i in np.arange(len(stages)):
        s = stages[i]
        label = labeldict.get(s)
        if label is None:
            p = 99
            if '??' not in labels:
                labels.append('??')
        else:
            p = -labels.index(label)

        # make some red line markers for REM, mark beginning and end of REM
        if 'REM' in labels:
            if label == 'REM' and len(rem_start) == len(rem_end):
                rem_start.append(i-2)
            elif label != 'REM' and len(rem_start) > len(rem_end):
                rem_end.append(i-1)
        if label == 'REM' and i == len(stages)-1:
            rem_end.append(i+1)

        if i != 0:
            y.append(p)
            x.append(i-1)
        y.append(p)
        x.append(i)

    assert len(rem_start) == len(
        rem_end), 'Something went wrong in REM length calculation'

    x = np.array(x)*epochlen
    y = np.array(y)
    y[y == 99] = y.min()-1  # make sure Unknown stage is plotted below all else

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure
    timeoffset = 0 if starttime is None else (starttime.hour*60*60+starttime.minute*60)//300*300
    formatter = matplotlib.ticker.FuncFormatter(
        lambda s, x: time.strftime('%H:%M', time.gmtime(s+timeoffset)))

    ax.plot(x, y, **kwargs)
    ax.set_xlim(0, x[-1])
    ax.xaxis.set_major_formatter(formatter)

    ax.set_yticks(np.arange(len(np.unique(labels)))*-1)
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(0, x[-1], 3600))
    if xlabel:
        if starttime:
            ax.set_xlabel('Time of day')
        else:
            ax.set_xlabel('Time elapsed after recording start')
    if ylabel:
        ax.set_ylabel('Sleep Stage')
    if title is not None:
        ax.set_title(title)

    try:
        warnings.filterwarnings(
            "ignore", message='This figure includes Axes that are not compatible')
        fig.tight_layout()
    except Exception:
        pass

    # plot REM in RED here
    for start, end in zip(rem_start, rem_end):
        height = -labels.index('REM')
        ax.hlines(height, start*epochlen, end*epochlen, color='r',
                  linewidth=4, zorder=99)


def plot_hypnogram_overview(hypnos_list, ax=None, cbar=True):
    """plot percentage of participants in certain sleep stage

    :param hypnos_list: list of hypnograms
    :type hypnos_list: list
    :param ax: ax to plot into
    :type ax: plt.Axes
    :return: the resulting image
    :rtype: np.ndarray

    """
    size = (len(hypnos_list), max([len(x) for x in hypnos_list]))
    hypnos = np.full(size, np.nan)

    for i, hypno in enumerate(hypnos_list):
        hypnos[i, :len(hypno)] = hypno

    hypno_map = np.full([100, hypnos.shape[-1]],
                        np.nanmax(hypnos)+1, dtype=int)
    pos_map = {0: 3, 1: 2, 2: 1, 3: 4, 4: 0, 5: np.nan}

    for t, stages in enumerate(hypnos.T):
        s, counts = np.unique(stages, return_counts=True)
        perc = dict(zip(s, [int(c/sum(counts)*100) for c in counts]))

        psum = 0
        for pos, stage in pos_map.items():
            p = perc.get(stage)
            if p is not None:
                hypno_map[psum:psum+p, t] = -1 if np.isnan(stage) else stage
                psum += p

    if ax is None:
        ax = plt.gca()

    cmap = ListedColormap(
        ['blue', 'lightblue', 'lightgreen', 'darkgreen', 'red', 'gray'][:len(np.unique(hypno_map))])
    assert len(cmap.colors) == len(np.unique(hypno_map)
                                   ), 'must have same number of colors and stages'
    heatmap = ax.imshow(np.flipud(hypno_map),  aspect='auto',
                        interpolation='nearest', cmap=cmap, vmin=0, vmax=len(cmap.colors))

    ax.set_title(
        'Distribution of sleep stages after recording start for all participants')
    ax.set_xlabel('Time after recording start')
    ax.set_ylabel('% of participants in stage')

    positions = [m for m in range(0, hypno_map.shape[-1], 120)]
    ticklabel = [f'{m//2//60}:00' for m in positions]
    plt.xticks(positions, ticklabel)

    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size='10%', pad=0.6)
        ax.figure.colorbar(heatmap, cax=cax, orientation='horizontal')

        for i, text in enumerate(['Wake', 'N1', 'N2', 'SWS', 'REM', 'NaN']):
            c = 'white' if i in [0, 3] else None
            cax.text(i+0.5, 0.5, text, ha="center", va="center", color=c)
    return heatmap


def specgram_matplotlib(data, sfreq, NFFT=1500, ax=None):
    """
    Wrapper to nicely visualize the spectogram of EEG using matplotlib.specgram
    """
    if ax is None:
        plt.figure()
        ax = plt.subplot(1, 1, 1)

    raw = data.squeeze()
    assert raw.ndim == 1, 'Data must only have one dimension'

    ax.specgram(raw, Fs=sfreq, NFFT=NFFT, noverlap=sfreq, scale='dB')
    ax.set_ylim(0, 50)

    formatter = matplotlib.ticker.FuncFormatter(
        lambda s, x: time.strftime('%H:%M', time.gmtime(s)))
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xticks(np.arange(0, len(raw)//sfreq, 3600))

    ax.set_ylabel('Frequency')
    ax.set_xlabel('Time after onset')
#    plt.title('Spectorgram (Fs {} Hz, avg {} seconds)'.format(sfreq, NFFT//sfreq))


def specgram_welch(data, sfreq, nfft=None, nperseg=None, noverlap=None,
                   window='hann', ax=None):
    """
    Display EEG spectogram using a multitaper from 0-30 Hz

    :param data: the data to visualize, should be of rank 1
    :param sfreq: the sampling frequency of the data
    :param sperseg: number of seconds to use per FFT
    :param noverlap: percentage of overlap between segments
    :param ax: an axis where to plot. Else will create a new figure.
    """

    if ax is None:
        plt.figure()
        ax = plt.subplot(1, 1, 1)

    if nfft is None:
        nfft = int(sfreq*30)
    if nperseg is None:
        nperseg = int(sfreq*2)
    if noverlap is None:
        noverlap = int(sfreq*0.5)

    data_seg = data[:len(data)-len(data) % nfft]
    data_seg = data_seg.reshape([-1, nfft])

    freq, spec = welch(data_seg, fs=sfreq, window=window,
                       nperseg=nperseg, noverlap=noverlap)
    spec = np.log10(spec)*20

    f_lbound = np.abs(freq - 0.5).argmin()
    f_ubound = np.abs(freq - 35).argmin()
    spec = spec[:, f_lbound:f_ubound]
    freq = freq[f_lbound:f_ubound]

    ax.imshow(np.flipud(spec.T), aspect='auto')
    return spec.T


def plot_spectrogram_sensors(raw, ch_type='grad', decimate=100, block=True,
                             lfreq=0, ufreq=35, title=None, **kwargs):
    if isinstance(raw, str):
        raw = mne.io.read_raw(raw, )
    raw = raw.pick(ch_type)

    if title is None:
        try:
            title = os.path.basename(raw._filenames[0]) if len(raw._filenames) else 'unknown file'
        except (AttributeError, TypeError):
            title = 'unknown file'

    if decimate:
        data = raw.get_data()
        factor = np.round(raw.info['sfreq']//decimate).astype(int)
        data = data[:, ::factor]  # decimate data by subsampling, much faster
        info_decimated = raw.info.copy()
        with info_decimated._unlock():
            info_decimated['sfreq'] /= factor
        raw_decimated = mne.io.RawArray(data, info=info_decimated)
        raw = raw_decimated


    def plot_spectrum_for_sensors(ch_selection):
        if not ch_selection:
            return
        fig, ax = plt.subplots(1, 1, figsize=[10, 6])
        data = raw.get_data(ch_selection).squeeze().mean(0)
        specgram_multitaper(data, sfreq=raw.info['sfreq'],
                            ufreq=ufreq, lfreq=lfreq, ax=ax, **kwargs)
        ch_selection_name = ch_selection if len(ch_selection)<10 else f'{len(ch_selection)} channels'

        ax.set_title(f'{title}\nAvg. of {ch_selection_name}')
        fig.tight_layout()

    fig, ax = plt.subplots(1, 1)
    fig, _ = raw.plot_sensors('select', ch_type=ch_type, show_names=True, block=False, axes=ax)
    plotting_lambda = lambda x: plot_spectrum_for_sensors(fig.lasso.selection)
    fig.canvas.mpl_connect('button_release_event', plotting_lambda)
    plot_spectrum_for_sensors(raw.ch_names)
    plt.show(block=block)

def specgram_multitaper(data, sfreq, sperseg=30, perc_overlap=1/3,
                        lfreq=0, ufreq=60, show_plot=True, ax=None,
                        title=None, annotations=None, **kwargs):
    """
    Display EEG spectogram using a multitaper from 0-30 Hz (default)



    :param data: the data to visualize, should be of rank 1
    :param sfreq: the sampling frequency of the data
    :param sperseg: number of seconds to use per FFT
    :param noverlap: percentage of overlap between segments
    :param lfreq: Lower frequency to display
    :param ufreq: Upper frequency to display
    :param show_plot: If false, only the mesh is returned, but not Figure opened
    :param ax: An axis where to plot. Else will create a new Figure
    :param annotations: a mne annotations object that will be plotted with red
    :returns: the resulting mesh as it would be plotted
    """
    if ax is None:
        plt.figure()
        ax = plt.subplot(1, 1, 1)

    data = data.squeeze()
    assert data.ndim==1, f'data must be one dimensional, but has {data.ndim=}'
    data = zscore(data)*5  # to get the scaling right

    assert isinstance(show_plot, bool), 'show_plot must be boolean'
    nperseg = int(round(sperseg * sfreq))
    overlap = int(round(perc_overlap * nperseg))

    f_range = [lfreq, ufreq]

    freqs, xy, mesh = spectrogram_lspopt(data, sfreq, nperseg=nperseg,
                                        noverlap=overlap, c_parameter=20.)
    if mesh.ndim == 3:
        mesh = mesh.squeeze().T
    mesh = 20 * np.log10(mesh+0.0000001)
    idx_notfinite = np.isfinite(mesh) == False
    mesh[idx_notfinite] = np.min(mesh[~idx_notfinite])

    f_range[1] = np.abs(freqs - ufreq).argmin()
    sls = slice(f_range[0], f_range[1] + 1)
    freqs = freqs[sls]


    mesh = mesh[sls, :]  # take frequencies of interest
    mesh = mesh - mesh.min()  # normalize from 0-1
    mesh = mesh / mesh.max()
    mesh = np.flipud(mesh)  # lower frequencies at bottom


    # now resize to a format in which we don't have to worry about resolution
    seconds = int(len(data)//sfreq)
    mesh = np.array(Image.fromarray(mesh).resize([seconds, mesh.shape[0]],
                                                 resample=Image.NEAREST))

    if show_plot:
        vmin = np.percentile(mesh, 2)
        vmax = np.percentile(mesh, 95)
        kwargs = {'cmap': 'viridis'} | kwargs
        ax.imshow(mesh, aspect='auto', vmin=vmin, vmax=vmax, **kwargs)

        t_fmt = '%M:%S' if seconds<60*15 else '%H:%M'
        formatter = matplotlib.ticker.FuncFormatter(lambda s, x: time.strftime(
            t_fmt, time.gmtime(int(s))))
        ax.xaxis.set_major_formatter(formatter)

        # probably need to adapt this to work with all parameters
        if xy[-1] <15: # 15 seconds
            tick_distance = 1  # per second
            fmt = 'seconds'
        elif xy[-1] < 60: # 1 minute
            tick_distance = 5  # per 5 seconds
            fmt = 'seconds'
        elif xy[-1] < 60*30: # 30 minutes
            tick_distance = 60  # per 1 min
            fmt = 'minutes'
        elif xy[-1] < 3600: # 1 hour
            tick_distance = 5*60  # per 5 min
            fmt = 'minutes'
        elif xy[-1] < 3600*7:  # 7 hours is half hourly
            tick_distance = 30*60 # per half hour
            fmt = 'hours'
        else:  # more than 7 hours hourly ticks
            tick_distance = 60*60  # plot hour
            fmt = 'hours'

        two_hz_pos = np.argmax(freqs > 1.99999999)
        ytick_pos = np.arange(0, len(freqs), two_hz_pos)
        ytick_label = np.linspace(
            ufreq, lfreq, len(ytick_pos)).round().astype(int)
        ax.set_xticks(np.arange(0, mesh.shape[1], tick_distance))
        ax.set_yticks(ytick_pos)
        ax.set_yticklabels(ytick_label)
        ax.set_xlabel(f'Time after onset ({fmt})')
        ax.set_ylabel('Frequency')
        warnings.filterwarnings(
            "ignore", message='This figure includes Axes that are not compatible')

        if annotations:
            for annot in annotations:
                onset = annot['onset']
                duration = annot['duration']
                desc = annot['description'].replace('Comment/', '')
                # dt = annot['orig_time']
                ax.axvspan(onset, onset+duration, alpha=0.2, color='red')
                ax.vlines(onset, *ax.get_ylim(), color='red', linewidth=0.2)
                ax.text(onset, -0.5, desc, color='red', horizontalalignment='right',
                        verticalalignment='top', rotation=90, alpha=0.75)

        if title is not None:
            ax.set_title(title)
        ax.figure.tight_layout()

    return mesh
