# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 16:41:13 2024

@author: simon.kern
"""
import os
import itertools
import tempfile
import mne
from functools import wraps

def tempfile_wrapper(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        try:
            tempfile_name = tempfile.NamedTemporaryFile().name + '.edf'
            res = func(*args, **kwargs, tmp_edf=tempfile_name)
        except Exception as e:
            try:
                os.remove(tempfile_name)
            except FileNotFoundError:
                print('temporary file not found, probably exception before')
            raise e
        return res
    return wrapped

@tempfile_wrapper
def predict_usleep_raw(raw, api_token, eeg_chs=None, eog_chs=None,
                   ch_groups=None, model='U-Sleep v2.0', saveto=None,  
                   seconds_per_label=30, tmp_edf=None):
    """convenience function to upload any mne.io.Raw to usleep
    will prepare the file by downsampling to 128 Hz and discarding
    any channels that are not used
    Parameters
    ----------
    edf_file : str
        link to an edf file
    eeg_chs : list
        list of channels that are of type EEG and should be used for prediction
            channel, e.g. [Fz, Cz]. ch_groups will be created based on that.
    eog_chs : list
        list of channels that are of type EOG and should be used for prediction
        channel, e.g. [lEOG, rEOG]. ch_groups will be created based on that.
    ch_groups : list
        list of channel tuple, where each tuple contains one EEG and one EOG 
        channel, e.g. [[Fz, lEOG], [Cz, lEOG], [Fz, rEOG], [Cz, lEOG]].
    api_token : str
        U-Sleep API token, apply for it at https://sleep.ai.ku.dk.
    model : str
        U-Sleep model to use, e.g. U-Sleep v1.0 or v2.0
    saveto : str, optional
        save hypnogram to this file, with one entry per second of 
        the hypnogram. The default is None.
    seconds_per_label : int
        number of seconds that each hypnogram label should span. default: 30

    Returns
    -------
    hypno : np.array
        list of hypnogram labels.
    """
    assert (eeg_chs is None == eog_chs is None) ^ (ch_groups is None), \
        'must either supply eeg_chs and eog_chs OR ch_groups'
    if eeg_chs is not None and eog_chs is not None and ch_groups is None:
        ch_groups = list(itertools.product(eeg_chs, eog_chs))
        
    raw = raw.copy()  # work on copy as we resample data etc.
    # convert to EDF file if not 
    print('converting file to EDF')
    chs = set([ch[0] for ch in ch_groups]) | set([ch[1] for ch in ch_groups])
    # chs_idx = [i for i, ch in enumerate(raw.ch_names) if ch in chs]
    # only keep channels that are actually requested
    if any([ch not in raw.ch_names for ch in chs]):
        raw.drop_channels([ch for ch in raw.ch_names if not ch in chs])

    # is resampled anyway internally, reduce data size
    if raw.info['sfreq']>128:
        print('downsampling to 128 hz')
        raw.resample(128, n_jobs=-2)  
        
    mne.export.export_raw(tmp_edf, raw, fmt='edf', overwrite=True)
    return predict_usleep(tmp_edf, api_token, eeg_chs=None, eog_chs=None,
                          ch_groups=ch_groups, model=model, saveto=saveto,  
                          seconds_per_label=seconds_per_label)

def predict_usleep(edf_file, api_token, eeg_chs=None, eog_chs=None,
                   ch_groups=None, model='U-Sleep v2.0', saveto=None,  
                   seconds_per_label=30):
    """helper function to retrieve a hypnogram prediction from usleep
    a valid API token is necessary to run the function.    

    Parameters
    ----------
    edf_file : str
        link to an edf file
    eeg_chs : list
        list of channels that are of type EEG and should be used for prediction
            channel, e.g. [Fz, Cz]. ch_groups will be created based on that.
    eog_chs : list
        list of channels that are of type EOG and should be used for prediction
        channel, e.g. [lEOG, rEOG]. ch_groups will be created based on that.
    ch_groups : list
        list of channel tuple, where each tuple contains one EEG and one EOG 
        channel, e.g. [[Fz, lEOG], [Cz, lEOG], [Fz, rEOG], [Cz, lEOG]].
    api_token : str
        U-Sleep API token, apply for it at https://sleep.ai.ku.dk.
    model : str
        U-Sleep model to use, e.g. U-Sleep v1.0 or v2.0
    saveto : str, optional
        save hypnogram to this file, with one entry per second of 
        the hypnogram. The default is None.
    seconds_per_label : int
        number of seconds that each hypnogram label should span. default: 30

    Returns
    -------
    hypno : np.array
        list of hypnogram labels.
    """
    from sleep_utils import write_hypno
    from usleep_api import USleepAPI

    # parameter checks
    assert 0<seconds_per_label
    assert isinstance(seconds_per_label, int), f'must be integer but is {seconds_per_label}'
    assert (eeg_chs is None == eog_chs is None) ^ (ch_groups is None), \
        'must either supply eeg_chs and eog_chs OR ch_groups'
        
    # Create an API object and (optionally) a new session.
    try:
        api = USleepAPI(api_token=api_token)
        assert api,  f'could init API: {api}, {api.content}'
    except ConnectionRefusedError as e:
        raise ConnectionRefusedError(f'{e.args}. Maybe your API token expired? Go to https://sleep.ai.ku.dk to get a new one')
    assert os.path.isfile(edf_file), f'{edf_file} not found'
    assert (session:=api.new_session(session_name=os.path.basename(edf_file))), f'could init session: {session}, {session.content}'

    if eeg_chs is not None and eog_chs is not None and ch_groups is None:
        ch_groups = list(itertools.product(eeg_chs, eog_chs))
    
    # See a list of valid models and set which model to use
    session.set_model(model)

    # Upload a local file (usually .edf format)
    print(f'uploading {edf_file}')
    assert (res:=session.upload_file(edf_file)), f'upload failed: {res}, {res.content}'
        
    # Start the prediction on two channel groups:
    #   1: EEG Fpz-Cz + EOG horizontal
    #   2: EEG Pz-Oz + EOG horizontal
    # Using 30 second windows (note: U-Slep v1.0 uses 128 Hz re-sampled signals)

    session.predict(data_per_prediction=128*seconds_per_label, 
                    channel_groups=ch_groups)

    # Wait for the job to finish or stream to the log output
    # session.stream_prediction_log()
    print('waiting for prediction')
    success = session.wait_for_completion()

    if success:
        # Fetch hypnogram
        hypno = session.get_hypnogram()['hypnogram']
        if saveto:
            write_hypno(hypno, saveto, mode='csv',
                        seconds_per_annotation=1, overwrite=True)
        # Download hypnogram file
        # session.download_hypnogram(out_path="./hypnogram", file_type="tsv")
    else:
        raise Exception(f"Prediction failed.\n\n{hypno}")

    # Delete session (i.e., uploaded file, prediction and logs)
    session.delete_session()
    return hypno