# -*- coding: utf-8 -*-
"""
Created on Mon May 23 10:52:38 2022

Find spindles using wonambi.detect.spindle.DetectSpindle
based on Mölle2011 - Mölle, M. et al. (2011) Sleep 34(10), 1411-21

@author: Simon Kern
"""

import os
os.environ['JOBLIB_CACHEDIR']='/data/joblib-sleep-utils/'
import yasa
import wonambi
import mne
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from wonambi.dataset import Dataset
import tempfile
from wonambi.trans import select, resample, frequency, get_times, fetch
from wonambi.detect import spindle
from sleep_utils import gui, tools, sigproc

#%% STATIC SETTINGS

# which method to use for spindle detection
# for a full list of methods see https://wonambi-python.github.io/gui/methods.html
# changing the method might need some more parameters to be adapted
# in the wonambi.detect.spindle.DetectSpindle method call below
method = 'Moelle2011'

# either use a fixed spindle peak, i.e. `13` Hz or calculate the individual
# spindle frequency using spectrogram peak estimation
spindle_peak = 'individual' # either a fixed number, ie. 13 or `individual`
spindle_range = 2 # +-2.5 Hz around the peak
spindle_duration = (0.5, 2) # min and max duration of spindles


#%% OPTIONS
files = gui.select_files()


common_chs = tools.get_common_channels(files)

# let user choose the channels to be used
recommended_eeg = ['z', 'c4', 'c3', 'f3', 'f4', 'p4', 'p3']
recommended_ref = ['M1', 'M2', 'A1', 'A2', 'Cz', 'Pz']

func = lambda x, prio: any([y in x.lower() for y in prio])
eogs = sorted(common_chs, key=lambda x:'AAA' if 'eog' in x.lower() else x)
eegs = sorted(common_chs, key=lambda x:'AAA' if func(x, recommended_eeg) else x)
refs = ['no re-referencing', 'average'] + sorted(common_chs, key=lambda x:'AAA' if func(x, recommended_ref) else x)

title = 'Select channels that will be used for spindle detection'
eeg_chs, ref_chs_orig = gui.display_listbox(lists=[eegs, refs],
                                                selected = [],
                                                labels=['EEG', 'Reference'],
                                                title=title)

stages = gui.display_listbox([['N1', 'N2', 'N3', 'REM', 'Wake']],
                             selected=['N2'],
                             title='Select stages for analysis')[0]

stages = [{'N1': 1, 'N2': 2, 'N3': 3, 'REM':4, 'Wake': 0}[s] for s in stages]


csv_file = gui.choose_file(default_file='spindles_summary.csv', exts='csv',
                            title='choose save file for results', mode='save')

#%% spindle detection

tmp_dir = tempfile.mkdtemp()

df = pd.DataFrame()

for file in (tqdm_loop:=tqdm(files)):
    basename = os.path.basename(file)
    tqdm_loop.set_description(f'{basename}: Loading data')

    raw = mne.io.read_raw(file, verbose='ERROR')
    raw.pick(eeg_chs + ([] if ref_chs_orig[0] in ['no re-referencing', 'average'] else ref_chs_orig))
    raw.load_data(verbose='ERROR')

    tqdm_loop.set_description(f'{basename}: Setting reference')
    if 'no re-referencing' in ref_chs_orig:
        ref_chs = []
    elif 'average' in ref_chs_orig:
        assert len(ref_chs)==1, f'If average referencing, must not select anything else {ref_chs=}'
        raw.set_eeg_reference('average')
        ref_chs = []
    else:
        assert len(ref_chs)>0, f'must select at least one reference or no reference {ref_chs=}'
        raw.set_eeg_reference(ref_chs)

    # now load the hypnogram that fits to this data
    hypno_file = tools.infer_hypno_file(file)
    hypno = tools.read_hypno(hypno_file, verbose=False)

    hypno_upsampled = yasa.hypno_upsample_to_data(hypno, sf_hypno=1/30, data=raw)
    assert len(raw)//raw.info['sfreq']//30==len(hypno)

    # load artefacts for this participant, if existing

    art_file = f'{file[:-4]}_artefacts.csv'
    try:
        artefacts = np.loadtxt(art_file).max(1)
        winlen = tools.get_var_from_comments(art_file, 'window_length')
    except FileNotFoundError:
        print(f'warning, no artefacts annotated for {file}, please run artefact detection script')
        artefacts = np.zeros_like(hypno)
        winlen = 30

    if spindle_peak=='individual':
        spindle_peak_freq = sigproc.get_individual_spindle_peak(raw, hypno_upsampled)
    else:
        spindle_peak_freq = spindle_peak

    tqdm_loop.set_description(f'{basename}: Determining segments')

    spindle_band = (spindle_peak_freq-spindle_range,
                    spindle_peak_freq+spindle_range)

    detector = wonambi.detect.spindle.DetectSpindle(method=method,
                                               frequency=spindle_band,
                                               duration=spindle_duration)
    dataset = Dataset(file)
    annot = tools.hypno2wonambi(hypno, artefacts, dataset, winlen=winlen)
    if artefacts.mean()>0.5:
        print(f'TOO MANY ARTEFACTS {basename}')
        continue

    df_all_spindles = pd.Series(dtype=float)
    df_mean = pd.DataFrame()

    for stage in stages:
        if not stage in hypno: continue
        segments = fetch(dataset, annot, cat=(1,1,1,0), stage=[str(stage)])

        segments.read_data(chan=eeg_chs, ref_chan=ref_chs)
        data = segments[0]['data']
        tqdm_loop.set_description(f'{basename}: Det, spindles stage {stage}')

        spindles = detector(data)

        df_spindles = pd.DataFrame(spindles.events)
        if len(df_spindles)==0:
            continue
        df_spindles['peak_val_orig'] = df_spindles['peak_val_orig'].astype(float)
        df_spindles['rms_orig'] = df_spindles['rms_orig'].astype(float)
        df_spindles['ptp_orig'] = df_spindles['ptp_orig'].astype(float)
        df_spindles['stage'] = stage
        df_all_spindles = pd.concat([df_all_spindles, df_spindles])

        mean_vars = ['dur', 'peak_val_det', 'peak_val_orig', 'auc_det', 'auc_orig',
                     'rms_det', 'rms_orig', 'power_orig', 'peak_freq', 'ptp_det',
                     'ptp_orig']

        df_mean_stage = df_spindles[mean_vars].mean()
        df_mean_stage['density'] = spindles.density[0]
        df_mean_stage['count'] = len(df_spindles)

        df_mean_stage.index += f'_stage{stage}'
        df_mean = pd.concat([df_mean, df_mean_stage], axis=0)

    df_mean = df_mean.transpose()
    df_mean.index = [basename]
    df_all_spindles.sort_values('start', inplace=True)
    df_all_spindles.to_csv(os.path.splitext(file)[0] + '_spindles.csv')

    df = pd.concat([df, df_mean])


df['method'] = method
df['spindle_peak'] = spindle_peak
df['spindle_range'] = str(spindle_range)
df['spindle_duration'] = str(spindle_duration)
df.to_csv(csv_file)
