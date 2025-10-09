# -*- coding: utf-8 -*-
"""
Created on Mon May 23 10:52:38 2022

Create HRV markers for the data

@author: Simon Kern
"""

import os
from scipy.misc import electrocardiogram
from sleepecg import detect_heartbeats
import sleep_utils
import numpy as np
import utils # must be executed in the same folder as the utils.py
from tqdm import tqdm
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import tempfile
import mne
import yasa
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from features import mean_HR, mean_RR, SDNN, RR_range, pNN50
from features import RMSSD, SDSD, LF_power, HF_power, LF_HF, SNSindex, PNSindex
from features import PermEn, ApEn, SD1, SD2
from features import rrHRV, HiguchiFract, PetrosianFract, KatzFract


#%% SETTINGS
# e.g. ['TV6', 'ZF6'],
# you can also specify to exclude all rest by e.g. ['REST']
# it will exclude all files that contain the strings within the quotation marks
exclude = ['']
data_dir = '/data/Exercise_Sleep_Project/ws22/'


eeg_ch = 'C4:M1'
ecg_ch = 'EKG II'

features = [mean_HR, mean_RR, SDNN, RR_range, pNN50, RMSSD, SDSD, LF_power,
            HF_power, LF_HF, SNSindex, PNSindex, PermEn, ApEn, SD1, SD2, 
            rrHRV, HiguchiFract, PetrosianFract, KatzFract]

#%% calculations

edf_files = utils.list_files(data_dir, ext='edf')

hypnos_rest = []
hypnos_ex = []

# first step: load all hypnograms
df_hrv = pd.DataFrame()

for edf_file in tqdm(edf_files, desc='calculating HRV'):
    
    # get some general information
    filename = os.path.basename(edf_file)
    subj, cond = utils.get_subj_cond(edf_file)

    # load the raw file
    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False, 
                              include=[eeg_ch, ecg_ch])
    
    eeg = raw.get_data(eeg_ch)
    ecg = raw.get_data(ecg_ch)
    
    # now load the hypnogram that fits to this data
    hypno_file = utils.infer_hypno_file(edf_file)
    hypno = sleep_utils.read_hypno(hypno_file, verbose=False)
    hypno_upsampled = yasa.hypno_upsample_to_data(hypno, sf_hypno=1/30, data=raw)
    sfreq = raw.info['sfreq']

    hrv, beats = yasa.hrv_stage(ecg, sfreq, threshold="5min", hypno=hypno_upsampled,
                            equal_length=True, rr_limit=(500, 1400))
    RR_windows = [np.diff(x)/sfreq for x in beats.values()]
    stages = [x[0] for x in beats.keys()]
    
    # sanity check, only include values with HR between 40 and 130
    include = (mean_HR(RR_windows)<130) & (40<mean_HR(RR_windows))
    
    df_feats = pd.DataFrame()
    for feat in features:
        val = np.array(feat(RR_windows))
        val[include] = np.nan
        df_tmp = pd.DataFrame({'value': val, 
                               'stage': stages,
                               'feat': feat.__name__,
                               'condition': cond,
                               'subj': subj})
        df_feats = pd.concat([df_feats, df_tmp], ignore_index=True)
    df_hrv = pd.concat([df_hrv, df_feats], ignore_index=True)    
stop

df_hrv=df_hrv.groupby(['condition', 'subj', 'stage', 'feat']).mean()
df_hrv.to_csv(f'{data_dir}/_results_HRV_simple.csv')
