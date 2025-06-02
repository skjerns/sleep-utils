# -*- coding: utf-8 -*-
"""
Created on Mon May 16 11:27:48 2022

This script performs some automatic artefact detection and saves
the results to the data directory as .txt files

@author: Simon
"""
import os
os.environ['JOBLIB_CACHEDIR']='/data/joblib-sleep-utils/'
import yasa
import mne
import scipy
import pandas as pd
import numpy as np
from tqdm import tqdm
from sleep_utils import gui, tools, sigproc, plotting
import matplotlib.pyplot as plt


#%% SETTINGS

files = gui.select_files(title='Select files for Sleep Scoring')

window_length = int(gui.display_textbox(label='Select window length in seconds', text='10'))

common_chs = tools.get_common_channels(files)

# let user choose the channels to be used
recommended_eeg = ['z', 'c4', 'c3', 'f3', 'f4', 'p4', 'p3']
recommended_ref = ['M1', 'M2', 'A1', 'A2', 'Cz', 'Pz']

func = lambda x, prio: any([y in x.lower() for y in prio])
eegs = sorted(common_chs, key=lambda x:'AAA' if func(x, recommended_eeg) else x)
refs = ['no re-referencing', 'average'] + sorted(common_chs, key=lambda x:'AAA' if func(x, recommended_ref) else x)

title = 'Select channels that will be used for prediction'
eeg_chs, ref_chs = gui.display_listbox(lists=[eegs, refs],
                                                selected = [],
                                                labels=['EEG', 'Reference'],
                                                title=title,
                                                mode='single')

#%% artefact detection
fig, ax = plt.subplots(1, 1, figsize=[14, 8])

df = pd.DataFrame()
for file in tqdm(files, desc='loading files'):

    plot_dir = os.path.dirname(file) + '/plots/'
    os.makedirs(plot_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(file))[0]

    # first of all, load the file itself into memory
    raw = sigproc.load_raw(file, sfreq=128, picks=eeg_chs)

    # secondly load the hypnogram associated with this sleep recording
    art = sigproc.artefact_heuristic(raw, wlen=window_length).astype(int)

    art_file = f'{file[:-4]}_artefacts.csv'
    comments = f'{window_length=}\n'
    comments += f'{eeg_chs=}\n'
    comments += f'{ref_chs=}\n'

    np.savetxt(art_file, art.max(1), fmt='%d', newline='\n', header=comments)


    # #%% save channel overview with noise levels
    # print('creating noise plot')
    # epoch_len = 10
    # noise_png = f'{plot_dir}/{basename}_noise.png'
    # picks = tools.filter_channels(eegs)
    # raw = sigproc.load_raw(file, picks=picks)
    # data, im = plotting.plot_noise_std(raw, ax=ax)
