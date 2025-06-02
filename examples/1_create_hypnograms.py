#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 10:21:01 2023

script to create hypnograms and sleep scoring for selected files
it is suggested to run the artefact detection beforehand

- can use YASA or USLEEP or dummy randomized sleep stages.
- will output a CSV file with summary statistics for each data file

@author: simon.kern
"""
import os
os.environ['JOBLIB_CACHEDIR']='/data/joblib-sleep-utils/'
import time
import mne
import numpy as np
import pandas as pd
import yasa
from tqdm import tqdm
import sleep_utils
from sleep_utils import gui, tools, usleep_utils, sigproc, plotting
import matplotlib.pyplot as plt

#%% STATIC SETTINGS
model = 'U-Sleep v2.0'
overwrite = False

#%% options to choose
files = gui.select_files(title='Select files for Sleep Scoring')

common_chs = tools.get_common_channels(files)

# let user choose the channels to be used
recommended_eeg = ['z', 'c4', 'c3', 'f3', 'f4', 'p4', 'p3']
recommended_ref = ['M1', 'M2', 'A1', 'A2', 'Cz', 'Pz']

func = lambda x, prio: any([y in x.lower() for y in prio])
eogs = sorted(common_chs, key=lambda x:'AAA' if 'eog' in x.lower() else x)
eegs = sorted(common_chs, key=lambda x:'AAA' if func(x, recommended_eeg) else x)
refs = ['no re-referencing', 'average'] + sorted(common_chs, key=lambda x:'AAA' if func(x, recommended_ref) else x)

method = gui.display_listbox([['u-sleep', 'yasa', 'random']],
                             labels=[['Choose a sleep scoring method']],
                             title='Choose sleep scoring algorith',
                             mode='single')[0][0].strip()

title = 'Select channels that will be used for prediction'
eeg_chs, eog_chs, ref_chs = gui.display_listbox(lists=[eegs, eogs, refs],
                                                selected = [],
                                                labels=['EEG', 'EOG', 'Reference'],
                                                title=title,
                                                mode='multiple' if method=='u-sleep' else 'single')


if method=='u-sleep':
    config = gui.config_load()
    prev_api_token = config.get('api_token', [])
    prev_api_token_time = config.get('api_token_time', 0)
    prev_dir = config.get('prev_dir', None)

    if time.time() - prev_api_token_time > 60*60*12:
        prev_api_token = ''

    api_token = gui.display_textbox(title='Enter API key',
                                label="Please enter U-Sleep API token obtained from https://sleep.ai.ku.dk\nThis key expires every 12 hours\nleave empty if you don't want to perform sleep staging",
                                text='').strip()
    config['api_token'] = api_token
    config['api_token_time'] = time.time()
    gui.config_save(config)

csv_file = gui.choose_file(default_file='hypno_summary.csv', exts='csv',
                           title='choose save file for results', mode='save')


#%% sleep score a hypnogram for each file
df = pd.DataFrame()

for file in (tqdm_loop:=tqdm(files, 'loading data')):

    basename = os.path.basename(file)

    # remove channels that were not selected

    plot_dir = f'{os.path.dirname(file)}/plots/'
    basename = os.path.basename(os.path.splitext(file)[0])
    os.makedirs(plot_dir, exist_ok=True)

    # predict hypnogram using chosen method
    hypno_file = os.path.splitext(file)[0] + '_hypno.txt'
    tqdm_loop.set_description('sleep scoring')

    if os.path.exists(hypno_file) and not overwrite:
        hypno = tools.read_hypno(hypno_file)
        print(f'Hypno file already exists, skip prediction: {hypno_file}')
    else:
        tqdm_loop.set_description(f'{basename}: Loading data')

        raw = sigproc.load_raw(file, sfreq=128, picks=eeg_chs+eog_chs,
                               filters=[0.1, 35])

        if method=='u-sleep':
            hypno = usleep_utils.predict_usleep_raw(raw, api_token,
                                                     eeg_chs=eeg_chs,
                                                     eog_chs=eog_chs,
                                                     model=model,
                                                     saveto=hypno_file)
        elif method=='yasa':
            sls = yasa.SleepStaging(raw, eeg_name=eeg_chs[0],
                                    eog_name=eog_chs[0])
            hypno_str = sls.predict()
            hypno = tools.transform_hypno(hypno_str, tools.conv_dict).astype(int)
            tools.write_hypno(hypno, hypno_file, mode='csv',
                              overwrite=True,
                              comment=f'YASA {yasa.__version__}')

        elif method=='random':
            n_epochs = int(np.round(raw.times.max()/30))
            hypno = tools.make_random_hypnogram(n_epochs)
            tools.write_hypno(hypno, hypno_file, mode='csv',
                              overwrite=True,
                              comment='RANDOM STAGES')

        else:
            raise Exception('unknown {method=}')

        hypno_png = f'{plot_dir}/{basename}_hypno.png'
        fig, ax = plt.subplots(1, 1, figsize=[10, 4])
        sleep_utils.plot_hypnogram(hypno, ax=ax)
        ax.set_title(f'Hypnogram for {basename} using {model}')
        plt.pause(0.1)
        fig.tight_layout()
        fig.savefig(hypno_png)
        plt.close(fig)


    # we load the file into memory using the function `mne.io.read_raw_edf`

    art_file = f'{file[:-4]}_artefacts.csv'
    try:
        art = np.loadtxt(art_file).max(1)
        winlen = tools.get_var_from_comments(art_file, 'window_length')
        art_chs = tools.get_var_from_comments(art_file, 'eeg_chs', typecast=str)
        art_ref = tools.get_var_from_comments(art_file, 'ref_chs', typecast=str)

    except (FileNotFoundError, np.AxisError):
        print(f'warning, no artefacts annotated for {file}, please run artefact detection script')
        art = np.zeros_like(hypno)
        winlen = 30
        art_chs = 'n/a'
        art_ref = 'n/a'

    hypno_art = yasa.hypno_upsample_to_data(hypno, sf_hypno=1/30,
                                            data=art, sf_data=1/winlen)
    hypno_art[art==-1]=1
    summary = sleep_utils.hypno_summary(hypno_art)

    # art = artefacts.max(1)
    art_total = np.sum(art) / len(art)*100
    art_wake = art[hypno_art==0].sum() / len(art[hypno_art==0])*100
    art_n1 = art[hypno_art==1].sum() / len(art[hypno_art==1])*100
    art_n2 = art[hypno_art==2].sum() / len(art[hypno_art==2])*100
    art_n3 = art[hypno_art==3].sum() / len(art[hypno_art==3])*100
    art_rem = art[hypno_art==4].sum() / len(art[hypno_art==4])*100
    summary |={'% art total': f'{art_total:.1f}%',
               '% art Wake': f'{art_wake:.1f}%',
               '% art N1': f'{art_n1:.1f}%',
               '% art N2': f'{art_n2:.1f}%',
               '% art N3': f'{art_n3:.1f}%',
               '% art REM': f'{art_rem:.1f}%',
               'artefact windows length': winlen,
               'artefact channels' : art_chs,
               'artefact references' : art_ref}

    df_subj = pd.DataFrame(summary, index=[basename])
    df = pd.concat([df, df_subj])

    ### plot noise as well
    # filter annotations that they are not overlapping in the plot
    tdiffs = np.pad(np.diff([annot['onset'] for annot in raw.annotations[1:]]).round(2), [0,1], constant_values=301)
    annotations = [annot for annot, tdiff in zip(raw.annotations[1:], tdiffs) if tdiff>300]

    fig, ax = plt.subplots(1, 1, figsize=[10, 4])
    ax2 = ax.twinx()
    for ch in tqdm(eeg_chs, desc='creating plot'):
        ax.clear()
        spect_png = f'{plot_dir}/{basename}_spect_{ch}.png'
        plotting.specgram_multitaper(raw.get_data(ch), raw.info['sfreq'], ax=ax, ufreq=30,
                            annotations=annotations)
        fig.suptitle(f'Spectrogram and hypnogram for {ch} for {os.path.basename(file)}')
        if hypno is not None:
            plotting.plot_hypnogram(hypno, ax=ax2, color='black', linewidth=0.5)

        plt.pause(0.1)
        fig.tight_layout()
        fig.savefig(spect_png)
    plt.close(fig)


df.to_csv(csv_file)
