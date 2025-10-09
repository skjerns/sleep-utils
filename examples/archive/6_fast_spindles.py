# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 09:37:49 2022

@author: Simon
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 23 10:52:38 2022

Find fast spindles using wonambi.detect.spindle.DetectSpindle
based on Mölle2011 - Mölle, M. et al. (2011) Sleep 34(10), 1411-21

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
import time
from wonambi.trans import select, resample, frequency, get_times, fetch
from wonambi.detect import spindle



def hypno2wonambi(hypno, artefacts, dataset):
    """
    create annotations file 
    
    :param hypno: hypnogram in 30 seconds base
    :param artefact: array with artefact markings
    :param dataset: a wonambi.Dataset type
    """
    # conv_dict = {0: 'Wake',
    #              1: 'NREM1',
    #              2: 'NREM2',
    #              3: 'NREM3',
    #              4: 'REM'}
    hypno_art = yasa.hypno_upsample_to_data(hypno, sf_hypno=1/30,
                                            data=artefacts, sf_data=1/winlen)
    tmp_xls = tmp_dir + f'/tmp_scoring_{subj}_{cond}_{time.time()}.xls'

    create_empty_annotations(tmp_xls, dataset)
    annot = Annotations(tmp_xls)
    annot.add_rater('U-Sleep')
    
    while len((stages:=annot.rater.find('stages'))) != 0:
        for stage in stages:
            stages.remove(stage)
            
    annot.create_epochs(winlen)
    
    assert len(hypno_art)==len(artefacts)

    for i, (stage, art) in enumerate(zip(hypno_art, artefacts)):
        # name = conv_dict[int(stage)]
        name = str(stage)
        if art:
            annot.set_stage_for_epoch(i*winlen, 'Poor',
                                                 attr='quality',
                                                 save=False)
        else:
            annot.set_stage_for_epoch(i*winlen, name, save=False)
        
    annot.save()
    return annot
#%% SETTINGS
data_dir = '/data/Exercise_Sleep_Project/ws22/'

channels = [ 'Pz'] # channel(s) that the spectrogram creation should be performed on
references = ['M1', 'M2'] # list of channels used to construct a reference

# denote stages that should be used for spindle detection
stages = [2, 3]

# which method to use for spindle detection
# for a full list of methods see https://wonambi-python.github.io/gui/methods.html
# changing the method might need some more parameters to be adapted
# in the wonambi.detect.spindle.DetectSpindle method call below
method = 'Moelle2011'

# use the fast spindle band only
spindle_bands = {'slow': (9, 12.5), 'fast':(13, 16)}
spindle_duration = (0.5, 2) # min and max duration of spindles


#%% calculations

edf_files = utils.list_files(data_dir, ext='edf')
tmp_dir = tempfile.mkdtemp()

tqdm_loop = tqdm(edf_files, total=len(edf_files)*len(spindle_bands)*len(stages))
df_summary = pd.DataFrame()

for edf_file in edf_files:
    subj, cond = utils.get_subj_cond(edf_file)
    tqdm_loop.set_description(f'{subj}_{cond}, Loading data')

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
    
    tqdm_loop.set_description(f'{subj}_{cond}, Determining segments')
    
    for band_name, spindle_band in spindle_bands.items():    
        detector = wonambi.detect.spindle.DetectSpindle(method=method, 
                                                   frequency=spindle_band,
                                                   duration=spindle_duration)
        
        dataset = Dataset(edf_file)  
        annot = hypno2wonambi(hypno, artefacts, dataset)
        
        df = pd.DataFrame()
        df_mean = pd.Series(dtype=float)
    
        for stage in stages:
            if not stage in hypno: continue
            segments = fetch(dataset, annot, cat=(1,1,1,0), stage=[str(stage)])
            
            segments.read_data(chan=channels, ref_chan=references)
            data = segments[0]['data']
            desc = f'{subj}_{cond}, Det, {band_name} spindles stage {stage}'
            tqdm_loop.set_description(desc)
        
            spindles = detector(data)
            
            df_spindles = pd.DataFrame(spindles.events)
        
            df_spindles['peak_val_orig'] = df_spindles['peak_val_orig'].astype(float)
            df_spindles['rms_orig'] = df_spindles['rms_orig'].astype(float)
            df_spindles['ptp_orig'] = df_spindles['ptp_orig'].astype(float)
            df_spindles['stage'] = stage
            df_spindles['band'] = str(spindle_band)
            df = pd.concat([df, df_spindles])
            
            mean_vars = ['dur', 'peak_val_det', 'peak_val_orig', 'auc_det', 'auc_orig',
                         'rms_det', 'rms_orig', 'power_orig', 'peak_freq', 'ptp_det',
                         'ptp_orig']
            df_mean_stage = df_spindles[mean_vars].mean()
            df_mean_stage['density'] = spindles.density[0]
            df_mean_stage['count'] = len(df_spindles)
            
            df_mean_stage.index += f'_stage{stage}'
            df_mean = pd.concat([df_mean, df_mean_stage], axis=0)
            tqdm_loop.update()

        df_mean.name = f'{subj}_{cond}_{band_name}'
        df_mean['subj'] = subj
        df_mean['condition'] = cond
        
        df_summary = df_summary.append(df_mean)
        
        spindles_csv = edf_file + f'_{band_name}_spindles.csv'
        df.sort_values('start', inplace=True)
        df.to_csv(spindles_csv)
    
df_summary['method'] = method    
df_summary['spindle_band'] = str(spindle_band)
df_summary['spindle_duration'] = str(spindle_duration)

cols = list(df_summary.columns)
cols = cols[-6:]+ cols[:-6]
df_summary = df_summary[cols]

summary_csv = f'{data_dir}/_results_slow-fast_spindles_RMS.csv'
df_summary.to_csv(summary_csv)

# fig, axs = plt.subplots(2, 1)

# sns.scatterplot(data=df_summary, x='subj', y='density_stage2', hue='condition', ax=axs[0])
# sns.scatterplot(data=df_summary, x='subj', y='density_stage3', hue='condition', ax=axs[1])
# axs[0].set_ylim([0.5, 2.5])
# axs[1].set_ylim([0.5, 2.5])
