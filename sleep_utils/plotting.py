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
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from lspopt import spectrogram_lspopt
from PIL import Image
import warnings
import seaborn as sns
import pandas as pd
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import zscore
import tkinter as tk
from tkinter import Tk, Listbox, Label
from tkinter.filedialog import askopenfilename, asksaveasfilename
from sleep_utils import usleep_utils



def choose_file(default_dir=None, default_file=None, exts='txt', 
                title='Choose file', mode='open'):
    """
    Open a file chooser dialoge with tkinter.
    
    :param default_dir: Where to open the dir, if set to None, will start at wdir
    :param exts: A string or list of strings with extensions etc: 'txt' or ['txt','csv']
    :returns: the chosen file
    """
    root = Tk()
    root.iconify()
    root.update()
    if isinstance(exts, str): exts = [exts]
    if mode=='open':
       name = askopenfilename(initialdir=default_dir,
                              default_file=default_file,
                              parent=root,
                              title = title,
                              filetypes =(*[("File", "*.{}".format(ext)) for ext in exts],
                                           ("All Files","*.*")))
    elif mode=='save':
        name = asksaveasfilename(initialdir=default_dir,
                              default_file=default_file,
                              parent=root,
                              title = title,
                              filetypes =(*[("File", "*.{}".format(ext)) for ext in exts],
                                         ("All Files","*.*")))
        if not name.endswith(exts[0]):
            name += f'.{exts[0]}'
    else:
        raise Exception(f'unknown mode: {mode}')
    root.update()
    root.destroy()
    if not os.path.exists(name):
        print("No file chosen")
    else:
        return name

def plotmulti(*args):
    """
    plots multiple signals, each signal in a new subplot.
    """
    plt.figure()
    for i, sig in enumerate(args):
        plt.subplot(len(args), 1, i+1)
        plt.plot(sig, linewidth=0.5)

def display_textbox(title='Enter text', label='Please enter text'):
    window = tk.Tk()
    window.title()
       
    # Label
    label = tk.Label(window, text=label)
    label.pack()
       
    # Text input field with line wrapping
    text_input = tk.Text(window, wrap=tk.WORD, height=5, width=50,)
    text_input.configure(font=("Arial", 8))  # Reducing font size

    text_input.pack(pady=5, padx=5)
       
    x = ['']
    # OK button
    def on_ok():
        # Get the text from the input field
        text = text_input.get("1.0", tk.END).strip()
        x[0] = text
        window.destroy()  # Close the window

    ok_button = tk.Button(window, text="OK", command=on_ok)
    ok_button.pack()

    # Run the Tkinter event loop
    window.mainloop()
    return x[0]


def display_listbox(items1, items2, title='select items'):
    # Create main tkinter window
    root = Tk()
    root.title(title)
    
    # Create labels
    label1 = Label(root, text="Select EEG")
    label1.grid(row=0, column=0)
    
    label2 = Label(root, text="Select EOG")
    label2.grid(row=0, column=1)
    
    # Create listboxes
    listbox1 = Listbox(root, selectmode=tk.EXTENDED, exportselection=False)
    listbox1.grid(row=1, column=0)
    for item in items1:
        listbox1.insert(tk.END, str(item))
    
    listbox2 = Listbox(root, selectmode=tk.EXTENDED,exportselection=False)
    listbox2.grid(row=1, column=1)
    for item in items2:
        listbox2.insert(tk.END, str(item))
        
    # Create scrollbars
    scrollbar1 = tk.Scrollbar(root, orient=tk.VERTICAL, command=listbox1.yview)
    scrollbar1.grid(row=1, column=0, sticky=tk.NS+tk.E)
    listbox1.config(yscrollcommand=scrollbar1.set)
    
    scrollbar2 = tk.Scrollbar(root, orient=tk.VERTICAL, command=listbox2.yview)
    scrollbar2.grid(row=1, column=1, sticky=tk.NS+tk.E)
    listbox2.config(yscrollcommand=scrollbar2.set)     
    # Bind listbox selection event to synchronize selection
 
    # Create OK button
    ok_button = tk.Button(root, text="OK", command=root.quit)
    ok_button.grid(row=2, columnspan=2, pady=10)
    
    # Run the tkinter event loop
    root.mainloop()
    
    selected_item_list1 = listbox1.curselection()
    selected_item_list2 = listbox2.curselection()
    
    selected_items1 = [listbox1.get(idx) for idx in selected_item_list1]
    selected_items2 = [listbox2.get(idx) for idx in selected_item_list2]
    root.destroy()
    
    return selected_items1, selected_items2

def create_psg_plot(api_token=None):
    if api_token is None:
        api_token = display_textbox(title='Enter API key',
            label="Please enter U-Sleep API token obtained from https://sleep.ai.ku.dk")
    file = choose_file(exts=['eeg', 'edf', 'fif', 'bdf'])
    if file.endswith('.eeg'):  # brainvision file is actually the header
        file = file[:-4] + '.vhdr'
    # load data header
    raw = mne.io.read_raw(file, preload=False)
    
    # let user choose the channels to be used
    recommended = ['z', 'c4', 'c3', 'f3', 'f4', 'p4', 'p3']
    func = lambda x: any([y in x.lower() for y in recommended])
    eogs = sorted(raw.ch_names, key=lambda x:'AAA' if 'eog' in x.lower() else x)
    eegs = sorted(raw.ch_names, key=lambda x:'AAA' if func(x) else x)
    title = 'Select channels that will be used for prediction'
    eeg_chs, eog_chs = display_listbox(eegs, eogs, title=title)
    
    # remove channels that were not selected
    raw.drop_channels([ch for ch in raw.ch_names if not ch in eeg_chs+eog_chs])
    if raw.info['sfreq']>128:
        print('downsampling to 128 hz')
        raw.resample(128, n_jobs=-1)
    raw.filter(0.1, 45, n_jobs=-1)
    
    hypno_file = os.path.splitext(file)[0] + '_hypno.txt'
    hypno = usleep_utils.predict_usleep_raw(raw, api_token, eeg_chs=eeg_chs,
                                            eog_chs=eog_chs, saveto=hypno_file)
    
    for ch in eeg_chs:
        png_file = os.path.splitext(file)[0] + f'_{ch}.png'
        fig, axs = plt.subplots(2, 1, figsize=[10, 8])
        plot_hypnogram(hypno, ax=axs[0])
        specgram_multitaper(raw.get_data(ch), raw.info['sfreq'], ax=axs[1], ufreq=30)
        fig.suptitle(f'Hypnogram and spectrogram for {ch} for {os.path.basename(file)}')
        plt.pause(0.1)
        fig.tight_layout()
        fig.savefig(png_file)
        

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
    for i, label in enumerate(target_names):
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


def plot_hypnogram(stages, labeldict=None, title=None, epochlen=30, ax=None,
                   verbose=True, xlabel=True, ylabel=True, **kwargs,):
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
        plt.figure()
        ax = matplotlib.pyplot.gca()
    formatter = matplotlib.ticker.FuncFormatter(
        lambda s, x: time.strftime('%H:%M', time.gmtime(s)))

    ax.plot(x, y, **kwargs)
    ax.set_xlim(0, x[-1])
    ax.xaxis.set_major_formatter(formatter)

    ax.set_yticks(np.arange(len(np.unique(labels)))*-1)
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(0, x[-1], 3600))
    if xlabel:
        plt.xlabel('Time after recording start')
    if ylabel:
        plt.ylabel('Sleep Stage')
    if title is not None:
        ax.set_title(title)

    try:
        warnings.filterwarnings(
            "ignore", message='This figure includes Axes that are not compatible')
        plt.tight_layout()
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
        ['blue', 'lightblue', 'lightgreen', 'darkgreen', 'red', 'gray'])
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


def specgram_multitaper(data, sfreq, sperseg=30, perc_overlap=1/3,
                        lfreq=0, ufreq=60, show_plot=True, ax=None,
                        title=None, annotations=None):
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
        ax.imshow(mesh, aspect='auto', cmap='viridis' , vmin=vmin, vmax=vmax)

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
