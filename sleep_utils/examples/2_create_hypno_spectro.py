# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:02:29 2024

This script will create take an EDF file, create USleep predictions and
spectrograms as pngs as well as plotting the variance of signals across time
as an indicator of signal quality

requirements:
    pip install edfio mne tqdm

@author: Simon.Kern
"""

import os
import mne
import json
import numpy as np
import edfio  # check if edfio is installed, necessary for usleep predict
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm
import time

from sleep_utils.tools import read_hypno
from sleep_utils.usleep_utils import predict_usleep_raw
from sleep_utils.external import appdirs
from sleep_utils.plotting import display_listbox, display_textbox, choose_file
from sleep_utils.plotting import plot_hypnogram, specgram_multitaper

config_dir = appdirs.user_config_dir('sleep-utils')
config_file = os.path.join(config_dir, 'last_used.json')
os.makedirs(config_dir, exist_ok=True)

def config_load():
    if not os.path.isfile(config_file):
        return {}
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def config_save(config):
    with open(config_file, 'w') as f:
        config = json.dump(config, f)

model = 'U-Sleep v2.0' # which usleep model to use

config = config_load()
sel_eeg = config.get('prev_eeg_selected', [])
sel_eog = config.get('prev_eog_selected', [])
sel_ref = config.get('prev_ref_selected', [])
prev_api_token = config.get('api_token', [])
prev_api_token_time = config.get('api_token_time', 0)
prev_dir = config.get('prev_dir', None)

if time.time() - prev_api_token_time > 60*60*12:
    prev_api_token = ''

#%% choose files to run analysis on
print('Please insert API in GUI loading..')
api_token = display_textbox(title='Enter API key',
                            label="Please enter U-Sleep API token obtained from https://sleep.ai.ku.dk\nThis key expires every 12 hours\nleave empty if you don't want to perform sleep staging",
                            text=prev_api_token).strip()

assert len(api_token) < 190, 'token might be too long? only paste once!'
config['api_token'] = api_token
config['api_token_time'] = time.time()
config_save(config)

print('Please select EDF file in GUI for loading..')
file = choose_file(default_dir = prev_dir, exts=['eeg', 'edf', 'fif', 'bdf', 'txt'], title='Choose file to analyze')

config['prev_dir'] = os.path.dirname(file)
config_save(config)

if file.endswith('.eeg'):  # brainvision file is actually the header
    file = file[:-4] + '.vhdr'

if file.endswith('.txt'):  # brainvision file is actually the header
    raise Exception(f'.txt files cannot be loaded and are only added to check which hypnogram has been created already')

# load data header
print(f'Loading data of {file}')
raw = mne.io.read_raw(file, preload=False)  # preload later, first show infoboxes

# let user choose the channels to be used
recommended_eeg = ['z', 'c4', 'c3', 'f3', 'f4', 'p4', 'p3'] + sel_eeg
recommended_ref = ['M1', 'M2', 'A1', 'A2', 'Cz', 'Pz']

func = lambda x, prio: any([y in x.lower() for y in prio])
eogs = sorted(raw.ch_names, key=lambda x:'AAA' if 'eog' in x.lower() or func(x, sel_eog) else x)
eegs = sorted(raw.ch_names, key=lambda x:'AAA' if func(x, recommended_eeg) else x)
refs = ['no re-referencing', 'average'] + sorted(raw.ch_names, key=lambda x:'AAA' if func(x, recommended_ref) else x)

title = 'Select channels that will be used for prediction'
eeg_chs, eog_chs, ref_chs = display_listbox(lists=[eegs, eogs, refs],
                                            selected = [sel_eeg, sel_eog, sel_ref],
                                            labels=['EEG', 'EOG', 'Reference'],
                                            title=title)

config['prev_eeg_selected'] = eeg_chs
config['prev_eog_selected'] = eog_chs
config['prev_ref_selected'] = ref_chs
config_save(config)

if not 'no re-referencing' in ref_chs:
    raw.set_eeg_reference(ref_chs)
    assert len(ref_chs)==1, f'If selecting no referencing, must not select anything else {ref_chs=}'
elif 'average' in ref_chs:
    raw.set_eeg_reference('average')
    assert len(ref_chs)==1, f'If average referencing, must not select anything else {ref_chs=}'

# remove channels that were not selected
print('loading raw data')
raw.load_data(verbose='INFO')
raw_orig = raw.copy()
raw.drop_channels([ch for ch in raw.ch_names if not ch in eeg_chs+eog_chs+ref_chs])
if raw.info['sfreq']>128:
    print('downsampling to 128 hz')
    raw.resample(128, n_jobs=-1, verbose='INFO')
raw.filter(0.1, 45, n_jobs=-1, verbose='INFO')

plot_dir = f'{os.path.dirname(file)}/plots/'
basename = os.path.basename(os.path.splitext(file)[0])
os.makedirs(plot_dir, exist_ok=True)

# predict hypnogram using usleep
hypno_file = os.path.splitext(file)[0] + '_hypno.txt'

if os.path.exists(hypno_file):
    hypno = read_hypno(hypno_file)
    print(f'Hypno file already exists, skip prediction: {hypno_file}')
elif api_token.strip()=='':
    print('No token inserted, skip prediction.')
    hypno = None
else:
    hypno = predict_usleep_raw(raw, api_token, eeg_chs=eeg_chs, eog_chs=eog_chs,
                               model=model, saveto=hypno_file)


#%% save hypnogram
if hypno is not None:
    hypno_png = f'{plot_dir}/{basename}_hypno.png'
    fig, ax = plt.subplots(1, 1, figsize=[10, 4])
    plot_hypnogram(hypno, ax=ax)
    ax.set_title(f'Hypnogram for {basename} using {model}')
    plt.pause(0.1)
    fig.tight_layout()
    fig.savefig(hypno_png)

#%% save channel overview with noise levels
print('creating noise plot')
epoch_len = 10
noise_png = f'{plot_dir}/{basename}_noise.png'
fig, ax = plt.subplots(1, 1, figsize=[14, 8])

data = raw_orig.get_data(picks='eeg')
sfreq = raw_orig.info['sfreq']

data = data[:, :int(len(raw)//(30*sfreq)*epoch_len*sfreq)]
data = data.reshape([len(data), -1, int(epoch_len*sfreq)])
stds = np.std(data, axis=-1) # get std per epoch as noise marker

# this is stupid as we are basically forcing 5% to be red
vmax = stds.ravel().mean()*3

ax.imshow(stds, cmap='RdYlGn_r', aspect='auto', interpolation='None', vmax=vmax)
ax.set_yticks(np.arange(len(data)), raw_orig.ch_names, fontsize=6)
# ax.set_xticks(np.arange(len(data)), raw_orig.ch_names, fontsize=6)
formatter = FuncFormatter(lambda x, pos: '{:.0f}'.format(x * epoch_len))
ax.xaxis.set_major_formatter(formatter)
ax.set_xlabel('Seconds')
ax.set_ylabel('Channels')
ax.set_title(f'Standard deviation of all channels {os.path.basename(file)} for {epoch_len} second segments')
plt.pause(0.1)
fig.tight_layout()
fig.savefig(noise_png)

#%% next create individual channel spectrogram plots

# filter annotations that they are not overlapping in the plot
tdiffs = np.pad(np.diff([annot['onset'] for annot in raw.annotations[1:]]).round(2), [0,1], constant_values=301)
annotations = [annot for annot, tdiff in zip(raw.annotations[1:], tdiffs) if tdiff>300]

fig, ax = plt.subplots(1, 1, figsize=[10, 4])
ax2 = ax.twinx()
for ch in tqdm(eeg_chs, desc='creating plot'):
    ax.clear()
    spect_png = f'{plot_dir}/{basename}_spect_{ch}.png'
    specgram_multitaper(raw.get_data(ch), raw.info['sfreq'], ax=ax, ufreq=30,
                        annotations=annotations)
    fig.suptitle(f'Spectrogram and hypnogram for {ch} for {os.path.basename(file)}')
    if hypno is not None:
        plot_hypnogram(hypno, ax=ax2, color='black', linewidth=0.5)

    plt.pause(0.1)
    fig.tight_layout()
    fig.savefig(spect_png)
