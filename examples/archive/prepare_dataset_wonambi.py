# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 12:22:11 2022

This script will prepare the EDF files such that they can be loaded into 
WONAMBI for further spindle analysis. 

It will do the following:
   - Filter signal between 0.1 and 30 Hz
   - Only keep channels that are needed for spindle analysis

@author: Simon Kern
"""
import os
import mne
import warnings
import sleep_utils
from tqdm import tqdm
from os.path import join
import os.path as op
import shutil
import mne
from mne_bids import write_raw_bids, BIDSPath, print_dir_tree
from mne_bids.stats import count_events


#%% SETTINGS

data_dir = 'Z:\\Exercise_Sleep_Project\\EDF Export EEG\\'

groups = ['EX', 'REST']

resample_to = 200 # resample to 200 Hz

filter_HP = 0.1 # filter below this
filter_LP = 30  # filter out above this

overwrite = True # overwrite if file exists?


#%% plot individual confmats
if __name__=='__main__':
    
    
    # list all EDF files in that folder
    files = os.listdir(data_dir)
    files = [file for file in files if file.endswith('.edf')]
    
    # create subfolder for groups
    for group in groups:
        os.makedirs(data_dir + f'/{group}', exist_ok=True)
    
    tqdm_loop = tqdm(total=len(files))
    
    
    # loop over all EDF files
    for file in files:
        
        curr_group = [g for g in groups if g in file]
        assert len(curr_group)==1, f'More than one matching group of filename {file} for groups {groups}'    
        curr_group = curr_group[0]
        
        raw = mne.io.read_raw_edf(join(data_dir, file))
        
        bids_root = data_dir + f'/bids-{curr_group}'
        subject_id = file[:3]
        task = file.split('_')[1]
        bids_path = BIDSPath(subject=subject_id, task=task, root=bids_root)
        write_raw_bids(raw, bids_path, overwrite=True)

        # group_dir = join(data_dir, curr_group)
        
        # hypno_file = file + '.txt'
        # if not os.path.exists(data_dir + f'/{hypno_file}'):
        #     warnings.warn(f'No hypnogram file named {hypno_file}, please check if it exists and is named correctly')
        
        # group = []
        
        # edf_new = join(group_dir , file.replace('.edf', '_prepared.edf'))
        # hypno_new = join(group_dir, hypno_file)

        # if os.path.exists(edf_new) and not overwrite: raise FileExistsError(f'File exists, no overwrite: {edf_new}')
        
        # tqdm_loop.set_description('loading edf')
        # include = ['EOGr:M1', 'EOGl:M2', 'EOGr:M2', 'C3:M2', 'C4:M1', 'F7', 'F8', 'Fz', 'Pz', 'M1', 'M2']
        # exclude = ['II', 'EOGl', 'EOGr', 'C3', 'C4', 'EKG II', 'EMG1', 'Akku', 
        #            'Akku Stan', 'Lage', 'Licht', 'Aktivitaet', 'SpO2', 'Pulse', 
        #            'Pleth', 'Flow&Snor', 'RIP Abdom', 'RIP Thora', 'Summe RIP', 
        #            'RIP', 'Stan', 'Abdom', 'Thora'] # already do not load these
        # raw = mne.io.read_raw_edf(join(data_dir, file), 
        #                           eog=['EOGr:M1',' EOGl:M2', 'EOGr:M2'],
        #                           exclude=exclude, preload=True, verbose=False)
        # raw.drop_channels([ch for ch in raw.ch_names if not ch in include])
        # tqdm_loop.set_description('filtering edf')
        # raw = raw.filter(filter_HP, filter_LP, verbose=False)   
        # tqdm_loop.set_description('resample edf')
        # raw = raw.resample(resample_to, verbose=False)
        # tqdm_loop.set_description('saving edf')
        # sleep_utils.write_mne_edf(raw, edf_new, overwrite=overwrite)
        
        # if os.path.exists(join(data_dir, hypno_file)):
        #     hypno = sleep_utils.read_hypno(join(data_dir, hypno_file))
        #     sleep_utils.write_hypno(hypno, hypno_new, mode='csv', overwrite=overwrite)
            
        # tqdm_loop.update()
        
    
