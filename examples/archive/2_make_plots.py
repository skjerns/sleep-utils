# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 07:12:51 2022

@author: Simon
"""
import os
import importlib
try:
    import utils
except ModuleNotFoundError:
    raise ModuleNotFoundError('Could not find utils script - please check you are' 
                              'running in the same folder as the spindle repository,'
                              ' and that utils.py is located there')


print('Loading libraries...')
utils.install('sleep-utils', ['--upgrade', '--no-deps'])

libraries = ['llvmlite', 'numpy', 'easygui', 'tqdm', 'pyedflib']
for lib in libraries:
    try:
        print(f'Loading {lib}...')
        importlib.import_module(lib)
    except ModuleNotFoundError:
        print(f'Library `{lib}` not found, attempting to install')
        utils.install(lib)
        
try:
    import mne
    import numpy as np
    import easygui
    from tqdm import tqdm
    from pyedflib import highlevel
    import matplotlib.pyplot as plt
    import sleep_utils
    from sleep_utils.sigproc import resample
except ModuleNotFoundError as e:
    
    raise ModuleNotFoundError(f'Could not find library {e.name}\nPlease install '
                              f'via the command `pip install {e.name}`')

import matplotlib.gridspec as gridspec
#%% SETTINGS

data_dir = '/data/Exercise_Sleep_Project/ws22/'

channels = ['Pz'] # channel(s) that the artefact detection should be performed on
references = ['M1', 'M2'] # list of channels used to construct a reference

#%% calculations


edf_files = utils.list_files(data_dir, ext='edf')

fig = plt.figure(figsize=[9,7])

gs = fig.add_gridspec(3,1) # two more for larger summary plots
axs = []

ax1 = fig.add_subplot(gs[:2, :])
ax2 = fig.add_subplot(gs[2:, :])
        
tqdm_loop = tqdm(total=len(edf_files), desc='creating spectrograms')

for edf_file in edf_files:
    ax1.clear()
    ax2.clear()
    subj = os.path.basename(edf_file)
    
    # we load the file into memory using the function `mne.io.read_raw_edf`
    tqdm_loop.set_description('Loading file')
    ####
    
    raw = utils.load_edf(edf_file, channels, references)

    # plot hypnogram

    hypno_file = utils.infer_hypno_file(edf_file)
    hypno = sleep_utils.read_hypno(hypno_file, verbose=False)
    sleep_utils.plot_hypnogram(hypno, ax=ax2, verbose=False) 
    

    # plot spectrogram
    sfreq = raw.info['sfreq']    
      
    tqdm_loop.set_description('Creating spectrogram')
    total_samples = int(len(hypno)*30*sfreq)
    data = raw.get_data(0).squeeze()*1e6 # convert to micro-Volt
    mesh = sleep_utils.specgram_multitaper(data, sfreq, ufreq=35, perc_overlap=0,
                                           ax=ax1, sperseg=30)
    
    # plot detected artefacts

    art_file = f'{edf_file[:-4]}_artefacts.csv'
    
    artefacts = np.loadtxt(art_file)
    artefacts = artefacts.max(1)    
    
    xspace = np.linspace(0, mesh.shape[-1]-1, len(artefacts)+1)    
    ax1.plot([xspace[0], xspace[-1]], [20, 20], c='white', linewidth=10)

    # first plot non-artefacts underneath
    for i, art in enumerate(artefacts):
        if art: continue
        ax1.plot([xspace[i], xspace[i+1]], [20, 20],'g', 
                     linewidth=10, alpha = 0.1)
    # then plot artefacted regions above
    for i, art in enumerate(artefacts):
        if not art: continue
        for x in range(36):
            ax1.plot([xspace[i], xspace[i+1]], [x, x], c='r', 
                 linewidth=3, alpha = 0.3)     
        
    ticks = ax1.get_yticklabels()    
    ticks[0] = 'ART'
    ax1.set_yticklabels(ticks)

    # save resulting figure
    png_file = edf_file + f'_spectrogram_{"-".join(channels)}.png'

    tqdm_loop.set_description('Saving plot')
    fig.suptitle(f'{subj} Channel: {channels} : {references}')
    plt.tight_layout()
    plt.pause(0.5)
    fig.savefig(png_file)
    tqdm_loop.update()
