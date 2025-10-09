
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 10:52:38 2022

First attempt to implement CVE as described in 
https://www.nature.com/articles/s41598-021-83817-6
(coefficient of variation of the envelope)


@author: Simon Kern
"""

import os
import yasa
import wonambi
import mne
import pandas as pd
import warnings
import sleep_utils
import numpy as np
import utils # must be executed in the same folder as the utils.py
from tqdm import tqdm
from datetime import datetime
import seaborn as sns
from scipy.signal import find_peaks, welch, detrend
import matplotlib.pyplot as plt
from wonambi.dataset import Dataset
from wonambi.attr import Annotations, create_empty_annotations
import tempfile
import scipy
from joblib import Parallel, delayed
from sleep_utils.sigproc import coeff_var_env


#%% SETTINGS
data_dir = '/data/Exercise_Sleep_Project/ws22/'


channels = ['C3'] # channel(s) that the spectrogram creation should be performed on
references = ['M2'] # list of channels used to construct a reference

# denote stages that should be used for spindle detection
stages = [2, 3]

band = (0.5, 4)

#%% calculations

edf_files = utils.list_files(data_dir, ext='edf')
tmp_dir = tempfile.mkdtemp()

df_summary = pd.DataFrame()

df_cves = pd.DataFrame()

for edf_file in tqdm(edf_files):
    subj, cond = utils.get_subj_cond(edf_file)

    raw = utils.load_edf(edf_file, channels, references)
    sfreq=raw.info['sfreq']
    
    # now load the hypnogram that fits to this data
    hypno_file = utils.infer_hypno_file(edf_file)
    hypno = sleep_utils.read_hypno(hypno_file, verbose=False)
    assert len(raw)//raw.info['sfreq']//30==len(hypno)

    # load artefacts for this participant
    art_file = f'{edf_file[:-4]}_artefacts.csv'
    artefacts = np.loadtxt(art_file).max(1)
    winlen = utils.get_var_from_comments(art_file, 'window_length')
    artefacts = artefacts.reshape([30//winlen, -1], order='F').sum(0)
    
    # remove artefacted epochs
    hypno = [h if a<1 else -1 for h, a in zip(hypno, artefacts) ]
    
    events = mne.make_fixed_length_events(raw, duration=30)
    epochs = mne.Epochs(raw, events, tmin=-18, tmax=48)    
       
    cve = Parallel(-1)(delayed(coeff_var_env)(sig, sfreq) for sig in epochs.get_data())

    # extend, as these are dropped by selecting ranges outside (i.e. -15)
    cve = [np.nan, *cve, np.nan]

    df = pd.DataFrame({'Epoch Nr.': np.arange(len(hypno)),
                       'Stage': hypno,
                       'CVE': cve,
                       'Condition': cond})
    
    df_cves = pd.concat([df_cves, df], ignore_index=True)
    
    cve_csv = edf_file + '_CVE.csv'
    df.to_csv(cve_csv)
    
    df_mean = df.groupby('Stage').mean()
    
    key = [f'{subj}_{cond}']
    _cvedata = {f'Stage {s}':df_mean['CVE'][s] for s in np.unique(hypno)}
    _cvedata['Condition'] = cond
  
    df_summary = pd.concat([df_summary, pd.DataFrame(_cvedata, index=key)])
    
    summary_csv = f'{data_dir}/_results_CVE.csv'

    df_summary.to_csv(summary_csv)


#%% plot results

fig, axs = plt.subplots(2, 3); axs=axs.flatten()

for i, s in enumerate(np.unique(np.arange(-1, 5))):
    ax = axs[i]
    sns.violinplot(data=df_summary, x='Condition', y=f'Stage {s}', ax=ax)
    
    x1 = df_summary[df_summary['Condition']=='REST'][f'Stage {s}']
    x2 = df_summary[df_summary['Condition']=='EX'][f'Stage {s}']
    
    p = scipy.stats.ttest_rel(x1, x2)
    
    ax.set_title(f'Stage {s}, p={p.pvalue:.3f}')

plt.tight_layout()


plt.figure()
bin_width = 60
mult = 1. / bin_width
df_cves['bin'] =np.floor(df_cves['Epoch Nr.'] * mult + .5) / mult
sns.lineplot(data=df_cves, x='bin', y='CVE', hue='Condition')
