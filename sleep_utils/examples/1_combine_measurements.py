#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 12:50:57 2024

script to merge several brainvision vhdr files

@author: simon.kern
"""
import os
import mne
import numpy as np
import json
import pybv
from tqdm import tqdm
from sleep_utils.plotting import choose_file
from sleep_utils.external import appdirs
import shutil

def common_prefix(strings):
    """find common prefix of strings, e.g flour, flower -> flo"""
    # Find the shortest string to limit the number of comparisons
    shortest = min(strings, key=len)
    for i in range(len(shortest)):
        # Check if this character is the same in all strings
        if any(s[i] != shortest[i] for s in strings):
            return shortest[:i]
    return shortest

def config_save(config):
    with open(config_file, 'w') as f:
        config = json.dump(config, f)

def config_load():
    if not os.path.isfile(config_file):
        return {}
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def sanitize(string):
    return ''.join([x if (x.isalnum() or x in '?.- ') else '-' for x in string])

config_dir = appdirs.user_config_dir('sleep-utils')
config_file = os.path.join(config_dir, 'last_used.json')
os.makedirs(config_dir, exist_ok=True)

config = config_load()
prev_dir = config.get('prev_dir', None)


# choose file
vhdr_files = choose_file(default_dir=prev_dir, exts=['vhdr'], multiple=True,
                         title='Choose files that should be combined')

config['prev_dir'] = os.path.dirname(vhdr_files[0])
config_save(config)

basename = common_prefix(vhdr_files)
if basename.endswith(('-', '_')):
    basename = basename[:-1]
print(f'Assuming the base name for this recording is {os.path.basename(basename)}')

all_raws = []
max_sfreq = 0

# load both files
raws = [mne.io.read_raw(file) for file in vhdr_files]
raws = sorted(raws, key=lambda x: x.info['meas_date'])
sfreq = raws[0].info['sfreq']
chs = raws[0].info['ch_names']

# check they are compatible
assert len(set([raw.info['sfreq'] for raw in raws]))==1, 'different sample frequencies!'
assert len(set([len(raw.info['ch_names']) for raw in raws]))==1, 'different number of channels!'
assert all([raw.info['ch_names']==raws[0].info['ch_names'] for raw in raws])==1, 'different names of channels!'
assert all([dt.total_seconds()<60*60*14 for dt in np.diff([raw.info['meas_date'] for raw in raws])]), f'Warning! Gap longer than 12 hours, sure these are the same recording? {[os.path.basename(file) for file in vhdr_files]}'

# calculate gap
gaps = []
samples = [len(raw) for raw in raws]

for i, raw in enumerate(raws[:-1]):
    raw_next = raws[i+1]
    gap = ((raw_next.info['meas_date']-raw.info['meas_date']).total_seconds())*sfreq-len(raw)
    print(f'gap {i+1} has length {gap//sfreq} seconds')
    gaps += [int(gap)]
gaps += [0]  # last recording has no gap obviously

# put data into one data file
data = np.zeros([len(chs), sum(gaps+samples)], dtype=np.float32)
annotations = []

offset = 0
for raw, gap in tqdm(zip(raws, gaps), total=len(raws), desc='loading data to concatenate'):
    data[:, offset:offset+len(raw)] = raw.get_data()

    for annot in raw.annotations:
        onset = annot['onset'] + offset/sfreq
        duration = annot['duration']
        description = sanitize(annot['description'])
        annotations.append([onset, duration, description])

    offset += gap+len(raw)

annotations = mne.Annotations(onset=[x[0] for x in annotations],
                              duration=[x[1] for x in annotations],
                              description=[x[2] for x in annotations])
# Create a new Raw object with the merged data
data = data.astype(np.float32)
merged_raw = mne.io.RawArray(data, raws[0].info.copy())
merged_raw.set_annotations(annotations)

# Write the merged data to new BrainVision files
print('writing new file')
vhdr_out = f'{basename}-combined-{len(raws)}files.vhdr'

# memory cleanup
del raws
del raw
del data

mne.export.export_raw(vhdr_out, merged_raw, fmt='brainvision')

# next move the 'old' files to a new directory

archive_dir = os.path.dirname(vhdr_files[0]) + '/single_files'
os.makedirs(archive_dir, exist_ok=True)

for vhdr in vhdr_files:
    basename = os.path.basename(vhdr)
    shutil.move(vhdr, os.path.join(archive_dir, basename))
    shutil.move(vhdr[:-4] + 'eeg', os.path.join(archive_dir, basename[:-4] + 'eeg'))
    shutil.move(vhdr[:-4] + 'vmrk', os.path.join(archive_dir, basename[:-4]+ 'vmrk'))
