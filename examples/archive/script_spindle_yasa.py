# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 17:39:57 2022

@author: Simon
"""

import mne # MNE is a package for processing EEG data in Python (e.g. loading EDF files)
import yasa # the package we previously installed, which contains most of the code we needy
import sleep_utils # this is a local package which is part of the GitHub repository
import itertools
from tqdm import tqdm

ch_include = ['C3:M2', 'C4:M1', 'Fz', 'Pz', 'M1', 'M2'] #only run analysis on these channels
ch_exclude = ['II', 'EOGl', 'EOGr', 'C3', 'C4', 'EKG II', 'EMG1', 'Akku', 
            'Akku Stan', 'Lage', 'Licht', 'Aktivitaet', 'SpO2', 'Pulse', 
            'Pleth', 'Flow&Snor', 'RIP Abdom', 'RIP Thora', 'Summe RIP', 
            'RIP', 'Stan', 'Abdom', 'Thora'] # already do not load these

data_dir = 'Z:/Exercise_Sleep_Project/EDF Export EEG/'# adapt the path on the left to where the data can be found on your computer
edf_file = data_dir + '/AA3_EX_AA3_EX_(1).edf'  # this is simply the first EDF file in the list

raw = mne.io.read_raw_edf(edf_file, exclude=ch_exclude, preload=True) # we load the file into memory using the function `mne.io.read_raw_edf`

# make sure the hypnogram of this participant is in the same folder as the EDF
hypnogram_file = edf_file + '.txt' # the hypnogram file
hypnogram = sleep_utils.read_hypno(hypnogram_file)
raw.drop_channels([ch for ch in raw.ch_names if not ch in ch_include])

raw = mne.set_bipolar_reference(raw, anode=['Fz','Fz', 'Pz', 'Pz'], cathode=['M1', 'M2', 'M1', 'M2'], verbose=False)

thresh = {'rel_pow': None, 'corr': 0.65, 'rms': 1.5}

params = itertools.product([250, 200, 100], [0.2, None], [0.65, None], [1.5, None])
res = {}

for sfreq, rel_pow, corr, rms in tqdm(list(params)):
    raw_res = raw.copy().pick('Pz-M2')
    raw_res.resample(sfreq)
    hypno = yasa.hypno_upsample_to_data(hypnogram, sf_hypno=1/30, data=raw_res)
    try:
        spindles = yasa.spindles_detect(raw_res, 
                                        hypno=hypno, 
                                        include=3,
                                        thresh = {'rel_pow': rel_pow, 'corr': corr, 'rms': rms},
                                        verbose='DEBUG') 
        res[f'{sfreq}, {str(rel_pow):>4}, {str(corr):>4}, {str(rms):>4}'] = len(spindles.summary())
    except:
        continue
#%%  
res = {}

HPs = [0, 0.1, 0.5, 1, 2]
LPs = [31, 40, 50, 75, 100]

for HP, LP in tqdm(list(itertools.product(HPs, LPs))):
    raw_res = raw.copy().pick('Pz-M2')
    raw_res.filter(HP, LP)
    hypno = yasa.hypno_upsample_to_data(hypnogram, sf_hypno=1/30, data=raw_res)
    spindles = yasa.spindles_detect(raw_res, 
                                    hypno=hypno, 
                                    include=3,
                                    verbose='DEBUG') 
    res[f'{str(HP):>3}, {str(LP):>3}'] = len(spindles.summary())
