# -*- coding: utf-8 -*-
"""
Created on Mon May 23 10:52:38 2022

Create timecourse of sleep stages for differen participants

@author: Simon Kern
"""

import os
import sleep_utils
import numpy as np
import utils # must be executed in the same folder as the utils.py
from tqdm import tqdm
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import tempfile



#%% SETTINGS
# e.g. ['TV6', 'ZF6'],
# you can also specify to exclude all rest by e.g. ['REST']
# it will exclude all files that contain the strings within the quotation marks
exclude = ['']
data_dir = '/data/Exercise_Sleep_Project/ws22/'


#%% calculations

edf_files = utils.list_files(data_dir, ext='edf')
tmp_dir = tempfile.mkdtemp()


hypnos_rest = []
hypnos_ex = []

# first step: load all hypnograms
for edf_file in tqdm(edf_files, desc='loading hypnograms'):
    filename = os.path.basename(edf_file)
    if any([(e in filename and not e=='') for e in exclude ]): continue
    subj, cond = utils.get_subj_cond(edf_file)

    # now load the hypnogram that fits to this data
    hypno_file = utils.infer_hypno_file(edf_file)
    hypno = sleep_utils.read_hypno(hypno_file, verbose=False)
    if '' in filename.upper():
        hypnos_rest.append(hypno)
    if 'EX' in filename.upper():
        hypnos_ex.append(hypno)
    

fig, axs = plt.subplots(3,1); axs=axs.flatten()
ax = axs[0]
sleep_utils.plot_hypnogram_overview(hypnos_rest, ax=ax, cbar=False)
ax.set_title(ax.get_title() + ' [NAP]')

ax = axs[1]
sleep_utils.plot_hypnogram_overview(hypnos_ex, ax=ax, cbar=False)
ax.set_title(ax.get_title() + ' [EX]')

ax = axs[2]
sleep_utils.plot_hypnogram_overview(hypnos_rest+hypnos_ex, ax=ax, cbar=True)
# ax.set_title(ax.get_title() + ' [ALL]', ax=ax)

fig.savefig(f'{data_dir}/_hypnogram_overview.png')
