# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 11:12:51 2021

@author: Simon
"""
import os
import ospath
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np
import mne
import pyedflib
from pyedflib import highlevel
import scipy
import seaborn as sns
from scipy.io import loadmat
import pandas as pd
import sleep_utils
import warnings;warnings.filterwarnings("ignore")
from usleep_api import USleepAPI
import itertools
from sklearn.metrics import f1_score, cohen_kappa_score, confusion_matrix
from tqdm import tqdm

#%%
def predict_usleep(edf_file, ch_groups, saveto=None):

    # Create an API token at https://sleep.ai.ku.dk.
    api_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE3MDg5ODEwODAsImlhdCI6MTcwODkzNzg4MCwibmJmIjoxNzA4OTM3ODgwLCJpZGVudGl0eSI6IjUzMTU5NGUwMTc0MSJ9.qm-1L6zTCuFfHvSFTI2uLGvE7jzZ7PLTS_V3piA58_4"  # Insert here

    # Create an API object and (optionally) a new session.
    api = USleepAPI(api_token=api_token)
    session = api.new_session(session_name=os.path.basename(edf_file))

    # See a list of valid models and set which model to use
    session.set_model('U-Sleep v2.0')

    # Upload a local file (usually .edf format)
    print(f'uploading {edf_file}')
    session.upload_file(edf_file)

    # Start the prediction on two channel groups:
    #   1: EEG Fpz-Cz + EOG horizontal
    #   2: EEG Pz-Oz + EOG horizontal
    # Using 30 second windows (note: U-Slep v1.0 uses 128 Hz re-sampled signals)


    session.predict(data_per_prediction=128*30, channel_groups=ch_groups)

    # Wait for the job to finish or stream to the log output
    # session.stream_prediction_log()
    print('waiting for prediction')
    success = session.wait_for_completion()

    if success:
        # Fetch hypnogram
        hypno = session.get_hypnogram()['hypnogram']
        if saveto:
            sleep_utils.write_hypno(hypno, saveto, mode='csv',
                                seconds_per_annotation=1, overwrite=True)
        # Download hypnogram file
        # session.download_hypnogram(out_path="./hypnogram", file_type="tsv")
    else:
        raise Exception(f"Prediction failed.\n\n{hypno}")

    # Delete session (i.e., uploaded file, prediction and logs)
    session.delete_session()
    return hypno


#%% plot individual confmats
if __name__=='__main__':

    files = ospath.list_files('Z:/Exercise_Sleep_Project/EDF Export EEG', exts='edf')
    files = [f for f in files if not '_filtered' in f]

    overwrite = False
    tqdm_loop = tqdm(total=len(files))

    for file in files:
        edf_new = file+'_filtered.edf'
        hypno_file = file + '.txt'

        if not os.path.exists(edf_new) or overwrite:
            tqdm_loop.set_description('loading edf')
            include = ['EOGr:M1', 'EOGl:M2', 'EOGr:M2', 'C3:M2', 'C4:M1', 'F7', 'F8', 'Fz', 'Pz', 'M1', 'M2']
            exclude = ['II', 'EOGl', 'EOGr', 'C3', 'C4', 'EKG II', 'EMG1', 'Akku',
                       'Akku Stan', 'Lage', 'Licht', 'Aktivitaet', 'SpO2', 'Pulse',
                       'Pleth', 'Flow&Snor', 'RIP Abdom', 'RIP Thora', 'Summe RIP',
                       'RIP', 'Stan', 'Abdom', 'Thora'] # already do not load these
            raw = mne.io.read_raw_edf(file, eog=['EOGr:M1',' EOGl:M2', 'EOGr:M2'], exclude =exclude, preload=True, verbose=False)
            raw.drop_channels([ch for ch in raw.ch_names if not ch in include])
            tqdm_loop.set_description('filtering edf')
            raw = mne.set_bipolar_reference(raw, anode=['F8', 'F7', 'Fz', 'Pz'], cathode=['M1', 'M2', 'M1', 'M2'], verbose=False)
            raw = raw.filter(0.1, 30, verbose='DEBUG')
            tqdm_loop.set_description('resample edf')
            raw = raw.resample(100, verbose=False)
            tqdm_loop.set_description('saving edf')
            sleep_utils.write_mne_edf(raw, file+'_filtered.edf', overwrite=True)

        if not os.path.exists(hypno_file):
            tqdm_loop.set_description('Uploading/predicting')
            eeg_chs = ['F7-M2', 'F8-M1', 'Fz-M1', 'Pz-M2', 'C4:M1', 'C3:M2']
            eog_chs = ['EOGr:M2', 'EOGr:M1', 'EOGl:M2']
            ch_groups = list(itertools.product(eeg_chs, eog_chs))
            hypno = predict_usleep(edf_new, ch_groups=ch_groups, saveto=hypno_file)
            sleep_utils.hypno_summary(hypno)

        tqdm_loop.update()
