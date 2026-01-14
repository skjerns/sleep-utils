# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 16:41:13 2024

@author: simon.kern
"""
import os
import io
import itertools
import tempfile
import mne
from functools import wraps
import numpy as np
import warnings
import pandas as pd
import requests
from io import BytesIO

def tempfile_wrapper(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        try:
            tempfile_name = tempfile.NamedTemporaryFile().name + '.edf'
            res = func(*args, **kwargs, tmp_edf=tempfile_name)
        finally:
            if os.path.isfile(tempfile_name):
                os.remove(tempfile_name)
        return res
    return wrapped

def disable_ssl_verify():
    """will monkey-patch requests made by usleep-api to veryify=False"""
    # Save original method
    from usleep_api import USleepAPI
    original_request = USleepAPI._request

    # Define patched method
    def patched_request(self, endpoint, method, as_json=False,
                        log_response=True, headers=None, **kwargs):
        kwargs.setdefault('verify', False)
        return original_request(self, endpoint, method, as_json=as_json,
                                log_response=log_response, headers=headers, **kwargs)

    # Apply monkey patch
    USleepAPI._request = patched_request
    warnings.warn('patched SSL to accept insecure connections')

def score_sleep(raw = None,
              edf_file = None, 
              api_token = None,
              backend = 'sleepyland',
              backend_url = None,
              eeg_chs=None,
              eog_chs=None,
              ch_groups=None,
              model=None,
              saveto=None,
              seconds_per_label=30,
              tmp_edf=None,
              return_proba=False):

    if (raw):
        return _score_sleep_raw(raw,
                api_token = api_token,
                backend = backend,
                backend_url = backend_url,
                eeg_chs = eeg_chs,
                eog_chs = eog_chs,
                ch_groups = ch_groups,
                model = model,
                saveto = saveto,
                seconds_per_label = seconds_per_label,
                return_proba = return_proba)
    elif (edf_file):
        return _score_sleep_file(edf_file,
                api_token = api_token,
                backend = backend,
                backend_url = backend_url,
                eeg_chs = eeg_chs,
                eog_chs = eog_chs,
                ch_groups = ch_groups,
                model = model,
                saveto = saveto,
                seconds_per_label = seconds_per_label,
                return_proba = return_proba)
    else:
        print("Requires either a valid EDF-file OR mne.io.Raw object")
    

@tempfile_wrapper
def _score_sleep_raw(raw, 
                    api_token = None,
                    backend = 'sleepyland',
                    backend_url = None, 
                    eeg_chs=None, 
                    eog_chs=None,
                    ch_groups=None, 
                    model='U-Sleep v2.0', 
                    saveto=None,
                    seconds_per_label=30, 
                    tmp_edf=None, 
                    return_proba=False):

    """
    Run U-Sleep prediction on an mne.io.Raw object.

    Prepares the raw data by selecting specified EEG/EOG channels,
    downsampling to 128 Hz, and exporting to EDF before submitting
    to the U-Sleep API.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG recording.
    api_token : str
        U-Sleep API token (https://sleep.ai.ku.dk).
    backend : str
        Which backend should be used for scoring.
    backend_url : str 
        URL for different backends. 
    eeg_chs : list of str, optional
        EEG channel names for prediction.
    eog_chs : list of str, optional
        EOG channel names for prediction.
    ch_groups : list of tuple, optional
        Pairs of (EEG, EOG) channels.
    model : str
        U-Sleep model version (default: 'U-Sleep v2.0').
    saveto : str, optional
        Path to save the predicted hypnogram.
    seconds_per_label : int
        Label duration in seconds (default: 30).
    tmp_edf : str, optional
        Optional path to use for temporary EDF export.
    return_proba : bool
        If True, return class probabilities.

    Returns
    -------
    np.ndarray or dict
        Hypnogram labels or probability predictions.
    """
    assert (eeg_chs is None == eog_chs is None) ^ (ch_groups is None), \
        'must either supply eeg_chs and eog_chs OR ch_groups'

    raw = raw.copy()  # work on copy as we resample data etc.
    # convert to EDF file if not
    print('converting file to EDF')
    if eeg_chs and eog_chs:
        chs = list(set(eeg_chs + eog_chs))
    elif ch_groups:
        chs = {channel for channel_value in ch_groups for channel in channel_value}
    # chs_idx = [i for i, ch in enumerate(raw.ch_names) if ch in chs]
    # only keep channels that are actually requested
#    if any([ch not in raw.ch_names for ch in chs]):
#        raw.drop_channels([ch for ch in raw.ch_names if not ch in chs])

    # is resampled anyway internally, reduce data size
    if raw.info['sfreq']>128:
        print('downsampling to 128 hz')
        raw.resample(128, n_jobs=-2)

    mne.export.export_raw(tmp_edf, raw, fmt='edf', overwrite=True)
    return _score_sleep_file(tmp_edf, 
                        api_token = api_token, 
                        backend = backend, 
                        backend_url = backend_url,
                        eeg_chs=eeg_chs, 
                        eog_chs=eog_chs,
                        ch_groups=ch_groups, 
                        model=model, 
                        saveto=saveto,
                        seconds_per_label=seconds_per_label,
                        return_proba=return_proba)

def delete_all_sessions(api_token):
    """convenience function to delete all sessions and data"""
    from usleep_api import USleepAPI
    api = USleepAPI(api_token=api_token)
    api.delete_all_sessions()

def _score_sleep_file(edf_file,
                api_token = None, 
                backend = 'sleepyland', 
                backend_url=None, 
                eeg_chs=None, 
                eog_chs=None,
                ch_groups=None, 
                model='U-Sleep v2.0', 
                saveto=None,
                seconds_per_label=30, 
                return_proba=False):
    """
    Run U-Sleep prediction on an EDF file via the U-Sleep API.

    Requires a valid API token. Optionally saves output and returns
    class probabilities.

    Parameters
    ----------
    edf_file : str
        Path to a local EDF file.
    backend : str
        Which backend should be used for scoring.
    backend_url : str 
        URL for different backends. 
    api_token : str
        U-Sleep API token (https://sleep.ai.ku.dk).
    eeg_chs : list of str, optional
        EEG channels used for prediction.
    eog_chs : list of str, optional
        EOG channels used for prediction.
    ch_groups : list of tuple, optional
        Explicit channel pairs (EEG, EOG). Overrides eeg_chs/eog_chs.
    model : str
        U-Sleep model version (default: 'U-Sleep v2.0').
    saveto : str, optional
        Path prefix for saving hypnogram (.csv) and confidences (.confidences.csv).
    seconds_per_label : int
        Seconds per hypnogram label (default: 30).
    return_proba : bool
        If True, also return class probability array.

    Returns
    -------
    hypno : np.ndarray
        Predicted hypnogram labels.
    proba : np.ndarray, optional
        Label probabilities (if return_proba is True).
    """
    from sleep_utils import write_hypno
#    from tools import write_hypno

#    if len(eeg_chs)==0 or len(eog_chs)==0:
#        raise ValueError('One element missing: {len(eeg_chs)=}, {len(eog_chs)=}')
    assert 0<seconds_per_label
    assert isinstance(seconds_per_label, int), f'must be integer but is {seconds_per_label}'
    assert (eeg_chs is None == eog_chs is None) ^ (ch_groups is None), \
        'must either supply eeg_chs and eog_chs OR ch_groups'

#check if file exists
    assert os.path.exists(edf_file), f"Error: '{edf_file}' does not exist."

#check if it is indeed an edf file
    assert os.path.splitext(edf_file)[1].lower() == '.edf', f"File '{edf_file}' does not have an '.edf' extension."

    try: 
        raw = mne.io.read_raw_edf(edf_file)
    except Exception as e:
        assert False, f"Error: File '{edf_file}' is not a valid EDF file. Details: {str(e)}"

#create a list of all requested channels
    if (eeg_chs and eog_chs):
        channels = eeg_chs + eog_chs
    elif (ch_groups):
        channels = {channel for channel_value in ch_groups for channel in channel_value}

#load channels from file
    all_channels = raw.ch_names

#check if all channels are present in file
    for i in channels:
        assert i in all_channels, f"Error: {i} not a channel in file"

    if (backend == 'sleepyland' or backend == 'sleepyland.zi.local' or backend == 'local'):
        if (return_proba):
            hypno, proba = _score_sleepyland(edf_file, eeg_chs=eeg_chs, eog_chs=eog_chs, ch_groups=ch_groups, return_proba=return_proba)
        else:
            hypno = _score_sleepyland(edf_file, eeg_chs=eeg_chs, eog_chs=eog_chs, ch_groups=ch_groups, return_proba=return_proba)

    elif (backend == 'denmark'):
        if (return_proba):
            hypno, proba = _score_usleep_denmark(edf_file, api_token, eeg_chs=eeg_chs, eog_chs=eog_chs, ch_groups=ch_groups, return_proba=return_proba)
        else:
            hypno = _score_usleep_denmark(edf_file, api_token, eeg_chs=eeg_chs, eog_chs=eog_chs, ch_groups=ch_groups, return_proba=return_proba)
        


    if saveto:
        write_hypno(hypno, saveto + '.csv', mode='csv',
                    seconds_per_annotation=1, overwrite=True)
        if return_proba:
#            header = ', '.join(classes)
            np.savetxt(saveto + '.confidences.csv', proba, fmt='%.8f',
                   delimiter=', ')
#                   header=header, delimiter=', ')

            # Download hypnogram file

        # Delete session (i.e., uploaded file, prediction and logs)
    return (hypno, proba) if return_proba else hypno

def _score_usleep_denmark(edf_file, 
                    api_token, 
                    eeg_chs=None, 
                    eog_chs=None,
                    ch_groups=None, 
                    model='U-Sleep v2.0',
                    seconds_per_label=30, 
                    return_proba=False):
    try:
        from usleep_api import USleepAPI
    except ModuleNotFoundError as e:
        raise(ModuleNotFoundError(f"{e}\n If missing, please install via 'pip install usleep_api --no-deps'"))    # parameter checks
    # Create an API object and a new session.
    try:
        api = USleepAPI(api_token=api_token)
        assert api,  f'could init API: {api}, {api.content}'
    except ConnectionRefusedError as e:
        raise ConnectionRefusedError(f'{e.args}. Maybe your API token expired? Go to https://sleep.ai.ku.dk to get a new one')
    assert os.path.isfile(edf_file), f'{edf_file} not found'

    with api.new_session_context(session_name=os.path.basename(edf_file)) as session:

        if eeg_chs is not None and eog_chs is not None and ch_groups is None:
            ch_groups = list(itertools.product(eeg_chs, eog_chs))

        assert len(ch_groups)<=24, f'EEG * EOG must be at maximum 24 combinations, but is {len(ch_groups)=}'

        # See a list of valid models and set which model to use
        session.set_model(model)

        # Upload a local file (usually .edf format)
        print(f'uploading {edf_file}')
        assert (res:=session.upload_file(edf_file)), f'upload failed: {res}, {res.content}'

        # Start the prediction on channel groups
        # Using 30 second windows (note: U-Slep v1.0 uses 128 Hz re-sampled signals)

        assert (res:=session.predict(data_per_prediction=128*seconds_per_label,
                 channel_groups=ch_groups)), f'prediction failed: {res}, {res.content}'

        # Wait for the job to finish or stream to the log output
        # session.stream_prediction_log()
        print('waiting for prediction')
        success = session.wait_for_completion()

        if success:
            # Fetch hypnogram
            with tempfile.TemporaryDirectory() as tmp:
                tmp = os.path.join(tmp, 'probas.npy')
                session.download_hypnogram(out_path=tmp, file_type='npy',
                                           with_confidence_scores=True)
                proba = np.load(tmp)

            res = session.get_hypnogram()
            hypno = res['hypnogram']
            classes = [res['classes'][k] for k in sorted(res['classes'])]


            # sanity check
            assert (proba.argmax(1)==hypno).all(), 'hypno from confidence is unequal hypno from get'
        else:
            raise Exception(f"Prediction failed.\n\n{hypno}")
    return (hypno, proba) if return_proba else hypno


def _score_sleepyland(filename, 
                    eeg_chs=None, 
                    eog_chs=None, 
                    ch_groups=None, 
                    model = 'usleep', 
                    return_proba=False):
#check if channels are set
    assert (eeg_chs and eog_chs) or ch_groups, f"Error: EEG and EOG channels or channels pairs must be set"


#create pairs
    if (eeg_chs and eog_chs):
        channels = eeg_chs + eog_chs
        pairs = ''
        for i in eeg_chs:
            for j in eog_chs:
                pair = i + '++' + j
                if pairs:
                    pairs = pairs + '&&' + pair
                else:
                    pairs = pair
    elif (ch_groups):
        channels = {channel for channel_value in ch_groups for channel in channel_value }
        pairs = '&&'.join([f"{a}++{b}" for a, b in ch_groups])
#setting variables
    host = 'sleepyland.zi.local'
    folder_name = os.path.splitext(os.path.basename(filename))[0]
    url = 'http://' + host + ':8887/process_one'


#prepare payload for POST request
    data = {
        'folderName': folder_name,
        'models': model,
        'channels': pairs,
    }
    files = {'edf-files': open(filename, 'rb')}

    print("waiting for prediction")

#Send the data
    try:
        response = requests.post(url, files=files, data=data)

        response.raise_for_status()

        print("Status Code:", response.status_code)
        print("Response:", response.text)

    except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")

    
#Download Data

    file = folder_name + '_PRED.npy' 

    api_url = 'http://' + host + ':8888/files/output/' + folder_name + '/usleep/majority/' + file
    result = requests.get(api_url)
    result.raise_for_status()  # This will raise an HTTPError for bad responses (4xx or 5xx)

#Create Probabilities

    proba = np.load(BytesIO(result.content))

    hypno = np.argmax(proba, axis=1).tolist()

    return (hypno, proba) if return_proba else hypno

