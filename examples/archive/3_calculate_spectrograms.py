# -*- coding: utf-8 -*-
"""
Created on Mon May 16 11:27:48 2022

This script calculates spectrograms for different sleep stages,
excluding the segments that were denoted has having artefacts

@author: Simon
"""
import os
import yasa
import mne
import pandas as pd
import sleep_utils
import numpy as np
import utils # must be executed in the same folder as the utils.py
from tqdm import tqdm

#%% SETTINGS

data_dir = 'Z:/Exercise_Sleep_Project/EDF Export EEG/'


window_length = 10 # time window for the spectrogram creation, e.g. 10 seconds

channels = ['Pz'] # channel(s) that the spectrogram creation should be performed on
references = ['M1', 'M2'] # list of channels used to construct a reference

# denote stages that should be used for spectrogram creation
stages = {'N2': [2],
          'N3': [3],
          'N2+N3': [2,3]}

max_freq = 35 # maximum frequency to calculate
freq_res = 0.5 # resolution of the frequency bands to use

#%%

edf_files = utils.list_files(data_dir, ext='edf')

tqdm_loop = tqdm(total=len(edf_files), desc='creating spectrograms')

df = pd.DataFrame()

for edf_file in edf_files:

    subj, cond = utils.get_subj_cond(edf_file)

    tqdm_loop.set_description('Loading file')

    # we load the file into memory using the function `utils.read_edf`
    raw = utils.load_edf(edf_file, channels, references)

    # now load the hypnogram that fits to this data
    hypno_file = utils.infer_hypno_file(edf_file)
    hypno = sleep_utils.read_hypno(hypno_file, verbose=False)
    hypno_upsampled = yasa.hypno_upsample_to_data(hypno, sf_hypno=1/30, data=raw)
    assert len(raw)//raw.info['sfreq']//30==len(hypno)

    # load artefacts for this participant
    art_file = f'{edf_file[:-4]}_artefacts.csv'
    artefacts = np.loadtxt(art_file).max(1)
    winlen = utils.get_var_from_comments(art_file, 'window_length')

    # upsample artefacts to match the hypnogram
    art_upsampled =  yasa.hypno_upsample_to_data(artefacts,
                                                 sf_hypno=1/winlen,
                                                 data=raw)

    # remove all artefacted segments from the hypnogram
    hypno_art = hypno_upsampled.copy()
    hypno_art[art_upsampled==1] = -1

    tqdm_loop.set_description('Calculating spectrogram')


    # go through the stages and stage combinations that we want
    res = pd.DataFrame({'Subject': subj,
                        'Condition': cond,
                        'winlen': [window_length],
                        'channel': str(channels),
                        'references': str(references)})

    for stage_name, stage in stages.items():
        if not stage in hypno_art: continue
        # create a temporary copy of the hypnogram, in which all
        # the stages of interest are marked as True, and all others as False
        hypno_stage = np.logical_or.reduce([hypno_art==s for s in stage])
        hypno_stage = [stage_name if h else '' for h in hypno_stage]

        bands = [(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
                (12, 16, 'Sigma'), (16, 30, 'Beta'), (30, 40, 'Gamma')]
        bands += [(freq, freq+freq_res, f'{freq}-{freq+freq_res}') for
                  freq in np.arange(0, max_freq, freq_res)]
        sp = yasa.bandpower(data=raw, hypno=hypno_stage, include=[stage_name],
                             win_sec=10, bands=bands, ch_names=channels)
        sp = sp.reset_index(level=[0, 1])
        sp = sp.drop(['Stage'], axis=1)
        sp.columns = [f'{stage_name}_{c}' for c in sp.columns]
        res = pd.concat([res, sp], axis=1)

    df = pd.concat([df, res], axis=0)
    tqdm_loop.update()

spectral_csv = f'{data_dir}/_results_spectral_power.csv'

df.to_csv(spectral_csv)
