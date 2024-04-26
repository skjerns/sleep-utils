# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 11:22:20 2018

@author: skjerns

This file contains all functions that deal with IO

"""
import os
import time
import warnings
import logging
import itertools
import tempfile
import mne
import numpy as np
from io import StringIO
from pprint import pprint
from datetime import datetime, timezone, timedelta


    
def sleep(seconds):
    if seconds > 1:
        for s in range(seconds):
            time.sleep(1)
    else:
        time.sleep(seconds)

def log(msg, *args, **kwargs):
    """
    Displays timestamp next to message
    """
    msg = '[{}] '.format(time.strftime('%H:%M:%S')) + msg
    print(msg, flush=True, *args, **kwargs)


def analyze_function(fun):
    """
    checks a couple of things for a function:
        is it scale invariant?
        is it ud/lr flip invariant?
    """
    signal = np.sin(np.linspace(0,100, 1000))
    out  = fun(signal)
    out2 = fun(signal*2)
    if np.allclose(out, out2):
        print('Scale invariant')
    elif np.allclose(out*2, out2):
        print('Preserves scale')
    else:
        print('Changes scaling')

    out  = fun(signal)
    out2 = fun(signal+2)
    if np.allclose(out, out2):
        print('Shift invariant')
    elif np.allclose(out+2, out2):
        print('Preserves shift')
    else:
        print('Changes shift')

def hypno_summary(hypno, epochlen=30, verbose=True):
    """
    param hypno: a hypnogram with stages in format
                 W:0, S1:1, S2:2, S3:3, REM:4
    param epochlen: the lenght of each entry in the hypnogram, e.g. 30 seconds

    summarizes the sleep parameters according to the AASM recommendations


        TST:     total sleep time - sum of minutes of sleep stages other than W
        TRT:     total recording time - duration from sleep onset to offset
        WASO:    total time spent in Wake between sleep onset/offset in minutes
        min S1:  total time spent in S1 in minutes
        min S2:  total time spent in S2 in minutes
        min S3:  total time spent in S3 in minutes
        min REM: total time spent in REM in minutes
        % WASO:  percentage Wake after sleep onset relative to TRT
        % S1:    relative time spent in S1 to TST
        % S2:    relative time spent in S2 to TST
        % S3:    relative time spent in S3 to TST
        % REM:   relative time spent in REM to TST
        lat S1:  latency of first S1 epoch after sleep onset in minutes
        lat S2:  latency of first S2 epoch after sleep onset in minutes
        lat S3:  latency of first S3 epoch after sleep onset in minutes
        lat REM: latency of first REM epoch after sleep onset in minutes

    For details and definitions see Iber et al (2007) The AASM Manual for the
    Scoring of Sleep and Associated Events.

    NB:
    sleep onset is defined by the AASM as the first non Wake epoch.
    Currently all parameters using lights out/lights on (eg. sleep efficiency)
    are not supported yet. Therefore, TRT will rely on sleep onset and sleep
    offset as markers for its calculation.


    """
    hypno = np.array(hypno)

    sleep_stages = {'W':0, 'S1':1, 'S2':2, 'SWS':3, 'REM':4}
    # do some sanity checks
    for stage, num in sleep_stages.items():
        if not num in hypno:
            warnings.warn(f'Stage {stage} not found in hypnogram')
    for stage in np.unique(hypno):
        if not stage in sleep_stages.values():
            warnings.warn(f'Found annotation with unknown value {stage}. '\
                          f'can only understand stages {sleep_stages}. '
                          'calculations will likely be wrong. Please '
                          'either transform to another stage (e.g. W) or'
                          ' remove from hypnogram.')

    onset = np.where(hypno!=0)[0][0] # first non-W epoch
    offset = np.where(hypno!=0)[0][-1] # last non-W epoch

    TST = (sum(hypno!=0)*epochlen)/60
    TRT = (offset-onset)*epochlen/60

    WASO = TRT-TST
    min_S1 = sum(hypno==1)*epochlen/60
    min_S2 = sum(hypno==2)*epochlen/60
    min_S3 = sum(hypno==3)*epochlen/60
    min_REM = sum(hypno==4)*epochlen/60

    perc_W = WASO/TRT
    perc_S1 = min_S1/TST
    perc_S2 = min_S2/TST
    perc_S3 = min_S3/TST
    perc_REM = min_REM/TST

    lat_S1 = (np.argmax(hypno==1)-onset)*epochlen/60
    lat_S2 = (np.argmax(hypno==2)-onset)*epochlen/60
    lat_S3 = (np.argmax(hypno==3)-onset)*epochlen/60
    lat_REM = (np.argmax(hypno==4)-onset)*epochlen/60

    sleep_onset_after_rec_start = onset*epochlen/60 # now convert to minutes
    sleep_offset_after_rec_start = offset*epochlen/60 # now convert to minutes
    recording_length = len(hypno)*epochlen/60

    summary = locals().copy()
    ignore = ['verbose', 'epochlen', 'stage', 'sleep_stages', 'num', 'hypno',
              'offset', 'onset']
    for var in ignore:
        del summary[var]

    if verbose: pprint(summary)

    return summary

def infer_hypno_file(filename):
    folder, filename = os.path.split(filename)
    possible_names = [filename + '.txt']
    possible_names += [filename + '.csv']
    possible_names += [os.path.splitext(filename)[0] + '.txt']
    possible_names += [os.path.splitext(filename)[0] + '.csv']
    possible_names += [os.path.splitext(filename)[0] + '_hypnogram.csv']
    possible_names += [os.path.splitext(filename)[0] + '_hypnogram.txt']
    possible_names += [os.path.splitext(filename)[0] + '_hypno.csv']
    possible_names += [os.path.splitext(filename)[0] + '_hypno.txt']

    for file in possible_names:
        if os.path.exists(os.path.join(folder, file)):
            return os.path.join(folder, file)
    warnings.warn(f'No Hypnogram found for {filename}, looked for: {possible_names}')
    return False


def read_hypno(hypno_file, epochlen = 30, epochlen_infile=None, mode='auto',
               exp_seconds=None, conf_dict=None, verbose=True):
    """
    reads a hypnogram file as created by VisBrain or as CSV type
    (or also some custom cases like the Dreams database)

    :param hypno_file: a path to the hypnogram
    :param epochlen: how many seconds per label in output
    :param epochlen_infile: how many seconds per label in original file
    :param mode: 'auto', 'time' or 'csv', see SleepDev/docs/hypnogram.md
    :param exp_seconds: How many seconds does the matching recording have?
    """
    assert str(type(epochlen)()) == '0'
    assert epochlen_infile is None or str(type(epochlen_infile)()) == '0'

    if isinstance(hypno_file, str):
        with open(hypno_file, 'r') as file:
            content = file.read()
            content = content.replace('\r', '') # remove windows style \r\n
    elif isinstance(hypno_file, StringIO):
        content = hypno_file.read()
        content = content.replace('\r', '') # remove windows style \r\n

    #conversion dictionary
    if conf_dict is None:
        conv_dict = {'W':0, 'WAKE':0, 'N1': 1, 'N2': 2, 'N3': 3, 'R':4, 'REM': 4, 'ART': 9,
                     -1:9, '-1':9, **{i:i for i in range(0, 10)}, **{f'{i}':i for i in range(0, 10)}}

    lines = content.split('\n')
    if mode=='auto':
        if lines[0].startswith('*'): # if there is a star, we assume it's the visbrain type
            mode = 'time'
        elif lines[0].replace('-', '')[0].isnumeric():
            mode = 'csv'
        elif lines[0].startswith('[HypnogramAASM]'):
            mode = 'dreams'
        elif lines[0].startswith(' Epoch Number ,Start Time ,Sleep Stage'):
            mode = 'alice'
        elif lines[0].startswith('EPOCH=') and lines[1].startswith('START='):
            mode = 'csv'
            lines = [l.upper() for l in lines[2:]]
        else:
            known_stage = [l.upper() in conv_dict for l in lines]
            if all(known_stage):
                mode='csv'
            else:
                mode=None

    if mode=='time':
        if epochlen_infile is not None:
            warnings.warn('epochlen_infile has been supplied, but hypnogram is time based,'
                          'will be ignored')
        stages = []
        prev_t = 0
        for line in lines:
            if len(line.strip())==0:   continue
            if line[0] in '*#%/\\"\'': continue # this line seems to be a comment
            s, t = line.split('\t')
            t = float(t)
            s = conv_dict[s]
            l = int(np.round((t-prev_t))) # length of this stage
            stages.extend([s]*l)
            prev_t = t

    elif mode=='csv':
        if exp_seconds and not epochlen_infile:
            epochlen_infile=exp_seconds//len(lines)
            if verbose: print('[INFO] Assuming csv annotations with one entry per {} seconds'.format(epochlen_infile))

        elif epochlen_infile is None:
            if len(lines) < 2400: # we assume no recording is longer than 20 hours
                epochlen_infile = 30
                if verbose: print('[INFO] Assuming csv annotations are per epoch')
            else:
                epochlen_infile = 1
                if verbose: print('[INFO] Assuming csv annotations are per second')
        lines = [line.split('\t')[0] if '\t' in line else line for line in lines]
        lines = [conv_dict[l]  if l in conv_dict else l for l in lines if len(l)>0]
        lines = [[line]*epochlen_infile for line in lines]
        stages = np.array([conv_dict[l] for l in np.array(lines).flatten()])

    # for the Dreams Database
    elif mode=='dreams':
        epochlen_infile = 5
        conv_dict = {-2:5,-1:5, 0:5, 1:3, 2:2, 3:1, 4:4, 5:0}
        lines = [[int(line)] for line in lines[1:] if len(line)>0]
        lines = [[line]*epochlen_infile for line in lines]
        stages = np.array([conv_dict[l] for l in np.array(lines).flatten()])

    elif mode=='alice':
        epochlen_infile = 30
        conv_dict = {'WK':0,'N1':1, 'N2':2, 'N3':3, 'REM':4}
        lines = [line.split(',')[-1] for line in lines[1:] if len(line)>0]
        lines = [[line]*epochlen_infile for line in lines]
        try: stages = np.array([conv_dict[l] for l in np.array(lines).flatten()])
        except KeyError as e:
            print('Unknown sleep stage in file')
            raise e
    else:
        raise ValueError('This is not a recognized hypnogram: {}'.format(hypno_file))

    stages = stages[::epochlen]
    if len(stages)==0:
        logging.warning('hypnogram loading failed, len == 0')

    return np.array(stages)


def hypno2time(hypno, seconds_per_epoch=1):
    """
    Converts a hypnogram into the format as defined by visbrain
    """
    hypno = np.repeat(hypno, seconds_per_epoch)
    s = '*Duration_sec {}\n'.format(len(hypno))
    stages = ['Wake', 'N1', 'N2', 'N3', 'REM', 'Art']
    d = dict(enumerate(stages))
    hypno_str = [d[h] for h in hypno]

    last_stage=hypno_str[0]

    for second, stage in enumerate(hypno_str):
        if stage!=last_stage:
            s += '{}\t{}\n'.format(last_stage, second)
            last_stage=stage
    s += '{}\t{}\n'.format(stage, second+1)
    return s

def write_hypno_time(hypno, filename, seconds_per_annotation=30, comment=None, overwrite=False):
    """
    Save hypnogram data with visbrain style
    The exact onset of each sleep stage is annotated in time space.
    This format is recommended for saving hypnograms

    :param filename: where to save the data
    :param hypno: The hypnogram either as list or np.array
    :param seconds_per_epoch: How many seconds each annotation contains
    :param overwrite: overwrite file?
    """
    assert not os.path.exists(filename) or overwrite, 'File already exists, no overwrite'
    hypno = np.repeat(hypno, seconds_per_annotation)
    hypno_str = hypno2time(hypno)
    if comment is not None:
        comment = comment.replace('\n', '\n*')
        hypno_str = '*' + comment + '\n' + hypno_str
        hypno_str = hypno_str.replace('\n\n', '\n')
    with open(filename, 'w') as f:
        f.write(hypno_str)
    return True


def write_hypno_csv(hypno, filename, seconds_per_annotation = 30, mode = 'seconds',
                    overwrite = False):
    """
    Save hypnogram data. Expects hypnogram to be based on epoch basis.
    it is saved as a csv-style file with one entry per second (default) and a new-line as separator

    :param filename: where to save the data
    :param hypno: The hypnogram either as list or np.array
    :param seconds_per_annotation: how many seconds does one annotation account for
    :param mode: 'seconds' or 'epochs':
                 'seconds' : writes one annotation per second
                 'epochs': write one annotation per 30 seconds
    :param overwrite: overwrite file?
    """
    assert not os.path.exists(filename) or overwrite, 'File already exists, no overwrite'
    assert mode in ['seconds', 'epochs'],'Mode must be seconds or epochs, is {}'.format(mode)
    hypno = np.array(hypno, copy=False)
    try:
        if np.any(np.logical_or(hypno>5, hypno<0)):
            raise ValueError('Contains values outside of [0, 5], which stage should that be?? ')
        with open(filename, 'w') as f:
            hypno_rep = [str(v) for v in np.repeat(hypno, seconds_per_annotation)]
            if mode=='epochs':
                hypno_rep = hypno_rep[::30]
            hypno_str = '\n'.join(hypno_rep)
            f.write(hypno_str)
    except Exception as e:
        print(e)
        return False
    return True


def write_hypno(hypno, filename, mode='time', seconds_per_annotation = 30, comment = None,
                overwrite = False):
    """
    Writes a hypnogram to file

        0 Wake
        1 N1
        2 N2
        3 N3
        4 REM
        5 Artefact / unknown

    :param filename: the filename under which to save a hypnogram
    :param hypno: a 1D array with sleep stage annotations.
    :param mode: 'time' or 'csv'
                 time: will save with the visbrain format (default)
                 csv: will save as a simple csv file
    :param seconds_per_annotation: how many seconds does one annotation account for
    :param comment: A comment that is added to the beginning if the file
    :param overwrite: Overwrite file if already present?
    """

    assert seconds_per_annotation>=1
    if mode=='time':
        if not filename.endswith('.hypno'):
            warnings.warn('Filename ends in ".{}", recommended to end in .hypno'.format(
                           os.path.splitext(filename)[1]))
        write_hypno_time(hypno, filename, seconds_per_annotation=seconds_per_annotation,
                         comment=comment, overwrite=overwrite)
    elif mode=='csv':
        write_hypno_csv(hypno, filename, seconds_per_annotation=seconds_per_annotation,
                        overwrite=overwrite)
    else:
        raise ValueError('Unkown mode {}, must be time or csv'.format(mode))




def transform_hypno(hypno, t_dict):
    """
    Will transform the hypnogram according to a lookup dictionary
    :param data: input data, will not be altered
    :param hypno: a hypnogram
    :param t_dict: a dictionary with lookup values.
                   if a value of hypno is not in t_dict it will not be altered
    :returns: data, new_hypnogram
    """
    hypno_values = np.unique(hypno)
    mapping = {h:h for h in hypno_values}
    mapping.update(t_dict)
    new_hypno = [mapping[h] for h in hypno]
    if type(hypno)==np.ndarray: new_hypno = np.array(new_hypno, dtype=hypno.dtype)
    return  new_hypno

def convert_hypnogram(hypno_file_in, hypno_file_out, **kwargs):
    """
    takes an arbitrary hypnogram and converts it to a time based format

    :param hypno_file_in:  A string pointing to a hypnogrma file
    :param hypno_file_out: Where the converted hypnogram should be saved
    """
    assert not os.path.exists(hypno_file_out)
    hypno = read_hypno(hypno_file_in, **kwargs)
    return write_hypno(hypno, hypno_file_out)


def _time_format(seconds):
    """ returns a string with a fitting string"""
    return '{:02d}:{:02d} hours'.format(seconds//3600, seconds%3600//60)


def _stamp_to_dt(utc_stamp):
    """Convert timestamp to datetime object in Windows-friendly way."""
    if 'datetime' in str(type(utc_stamp)): return utc_stamp
    # The min on windows is 86400
    stamp = [int(s) for s in utc_stamp]
    if len(stamp) == 1:  # In case there is no microseconds information
        stamp.append(0)
    return (datetime.fromtimestamp(0, tz=timezone.utc) +
            timedelta(0, stamp[0], stamp[1]))  # day, sec, μs


def write_mne_edf(mne_raw, fname, picks=None, tmin=0, tmax=None,
                  overwrite=False, verbose=False):
    """
    Saves the raw content of an MNE.io.Raw and its subclasses to
    a file using the EDF+/BDF filetype
    pyEDFlib is used to save the raw contents of the RawArray to disk
    Parameters
    ----------
    mne_raw : mne.io.Raw
        An object with super class mne.io.Raw that contains the data
        to save
    fname : string
        File name of the new dataset. This has to be a new filename
        unless data have been preloaded. Filenames should end with .edf
    picks : array-like of int | None
        Indices of channels to include. If None all channels are kept.
    tmin : float | None
        Time in seconds of first sample to save. If None first sample
        is used.
    tmax : float | None
        Time in seconds of last sample to save. If None last sample
        is used.
    overwrite : bool
        If True, the destination file (if it exists) will be overwritten.
        If False (default), an error will be raised if the file exists.
    """
    import pyedflib # pip install pyedflib
    from pyedflib import FILETYPE_BDF, FILETYPE_BDFPLUS, FILETYPE_EDF, FILETYPE_EDFPLUS
    if not issubclass(type(mne_raw), mne.io.BaseRaw):
        raise TypeError('Must be mne.io.Raw type')
    if not overwrite and os.path.exists(fname):
        raise OSError('File already exists. No overwrite.')

    # static settings
    has_annotations = True if len(mne_raw.annotations)>0 else False
    if os.path.splitext(fname)[-1] == '.edf':
        file_type = FILETYPE_EDFPLUS if has_annotations else FILETYPE_EDF
        dmin, dmax = -32768, 32767
    else:
        file_type = FILETYPE_BDFPLUS if has_annotations else FILETYPE_BDF
        dmin, dmax = -8388608, 8388607

    print('saving to {}, filetype {}'.format(fname, file_type))
    sfreq = mne_raw.info['sfreq']
    date = _stamp_to_dt(mne_raw.info['meas_date'])

    if tmin:
        date += timedelta(seconds=tmin)
    # no conversion necessary, as pyedflib can handle datetime.
    #date = date.strftime('%d %b %Y %H:%M:%S')
    first_sample = int(sfreq*tmin)
    last_sample  = int(sfreq*tmax) if tmax is not None else None


    # convert data
    channels = mne_raw.get_data(picks,
                                start = first_sample,
                                stop  = last_sample)

    # convert to microvolts to scale up precision
    eeg_chs = [ch_type=='eeg' for ch_type in mne_raw.get_channel_types()]
    channels[eeg_chs,:] *= 1e6

    # set conversion parameters
    n_channels = len(channels)

    # create channel from this
    try:
        f = pyedflib.EdfWriter(fname,
                               n_channels=n_channels,
                               file_type=file_type)

        channel_info = []

        ch_idx = range(n_channels) if picks is None else picks
        keys = list(mne_raw._orig_units.keys())
        for i in ch_idx:
            try:
                ch_dict = {'label': mne_raw.ch_names[i],
                           'dimension': mne_raw._orig_units[keys[i]],
                           'sample_rate': mne_raw._raw_extras[0]['n_samps'][i],
                           'physical_min': mne_raw._raw_extras[0]['physical_min'][i],
                           'physical_max': mne_raw._raw_extras[0]['physical_max'][i],
                           'digital_min':  mne_raw._raw_extras[0]['digital_min'][i],
                           'digital_max':  mne_raw._raw_extras[0]['digital_max'][i],
                           'transducer': '',
                           'prefilter': ''}
            except:
                ch_dict = {'label': mne_raw.ch_names[i],
                           'dimension': mne_raw._orig_units[keys[i]],
                           'sample_rate': sfreq,
                           'physical_min': channels[i].min(),
                           'physical_max': channels[i].max(),
                           'digital_min':  dmin,
                           'digital_max':  dmax,
                           'transducer': '',
                           'prefilter': ''}

            channel_info.append(ch_dict)

        subject_info = mne_raw._raw_extras[0].get('subject_info',{})
        f.setPatientCode(subject_info.get('id', '0'))
        f.setPatientName(subject_info.get('name', 'noname'))
        f.setTechnician('mne-gist-save-edf-skjerns')
        f.setSignalHeaders(channel_info)
        f.setStartdatetime(date)
        f.writeSamples(channels)

        for annotation in mne_raw.annotations:
            onset = annotation['onset']
            duration = annotation['duration']
            description = annotation['description']
            f.writeAnnotation(onset, duration, description)

    except Exception as e:
        raise e
    finally:
        f.close()
    return True
