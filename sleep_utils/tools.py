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
from joblib import Memory

conv_dict = {'W':0, 'WAKE':0, 'N1': 1, 'N2': 2, 'N3': 3, 'R':4, 'REM': 4, 'ART': 9,
             -1:9, '-1':9, **{i:i for i in range(0, 10)}, **{f'{i}':i for i in range(0, 10)}}


if (cachedir:=os.environ.get("JOBLIB_CACHEDIR")) is None:
    warnings.warn(
        "Environment variable JOBLIB_CACHEDIR is not defined. "
        "To enable caching, please set the env variable before importing "
        "sleep_utils via `import os; os.environ['JOBLIB_CACHEDIR']=xx`",
        stacklevel=2
    )

memory = Memory(cachedir)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def get_common_channels(files):
    """from a selection of MNE readable files, get the set of common channels"""
    from tqdm import tqdm
    chs = []
    for file in tqdm(files, desc='Scanning available channels'):
        raw = mne.io.read_raw(file, preload=False, verbose='ERROR')
        chs += [raw.ch_names]
    chs = [set(x) for x in chs]
    chs_common = set.intersection(*chs)
    return chs_common


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


def filter_channels(ch_names):
    """a heuristic function to filter channels that are not important

    will filter all light, battery, movement, temperature, breathing, audio,
    related channels
    """
    ch_names_filtered = []
    for ch in ch_names:
        if any(word in ch.lower() for word in ['light', 'acc', 'bat', 'temp', 'audio',
                                       'off', 'on', 'ssi', 'time', 'mic', 'adc',
                                       'act']):
            continue

        if len(ch) == 1:  # filter X, Y, Z
            continue

        ch_names_filtered += [ch]
    return ch_names_filtered


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

def hypno_summary(hypno, epochlen=30, lights_off_epoch=0, lights_on_epoch=-1,
                  print_summary=False, sanity_check=True):
    """ summarizes the sleep parameters according to the AASM recommendations

    It is assumed that the hypnogram is starting with the lights off marker and
    ending with the lights on marker. Otherwise, the epoch in which the night
    started and ended must be indicated.

        TST:     total sleep time - sum of minutes of sleep stages other than W
        TBT:     total bed time - duration from lights off to lights on
        TRT:     total recording time - from recording beginning to end
        SE:      sleep efficiency: TST/TBT
        WASO:    total time spent in Wake between sleep onset/offset in minutes
        min S1:  total time spent in S1 in minutes
        min S2:  total time spent in S2 in minutes
        min S3:  total time spent in S3 in minutes
        min REM: total time spent in REM in minutes
        total_NREM:     total NREM sleep time - sum of minutes in S1, S2, S3
        % WASO:  percentage Wake after sleep onset relative to time
                 between lights off and lights on (TBT)
        % S1:    relative time spent in S1 to all sleep stages (TST)
        % S2:    relative time spent in S2 to all sleep stages (TST)
        % S3:    relative time spent in S3 to all sleep stages (TST)
        % REM:   relative time spent in REM to TST
        lat S1:  latency of first S1 epoch after lights off in minutes
        lat S2:  latency of first S2 epoch after lights off in minutes
        lat S3:  latency of first S3 epoch after lights off in minutes
        lat REM: latency of first REM epoch after lights off in minutes
        awakenings:     number of awakenings
        dur_awakenings: average minutes of being awake
        FI:             fragmentation index - (Number of Awakenings + N
                                               umber of Stage Shifts) / TST
        sleep_cycles:   number of complete cycles (NREM to REM transitions)
        SQI:            sleep quality index - simplified as SE * (1 - FI)

    For details and definitions see Iber et al (2007) The AASM Manual for the
    Scoring of Sleep and Associated Events.

    :param hypno: a hypnogram in format W:0, S1:1, S2:2, S3:3, REM:4
    :param epochlen: the lenght of each entry in the hypnogram, e.g. 30 seconds
    :param lights_off_epoch: at which epoch lights were turned off
    :param lights_on_epoch: at which epoch lights were turned on
    :param print_summary: print output or not
    :param sanity_check: perform a sanity check on the values that have been
                         computed
    """
    s = AttrDict()
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
    # convert negative indices, e.g. -1 to their correct positive index
    lights_off_epoch = np.arange(len(hypno))[lights_off_epoch]
    lights_on_epoch = np.arange(len(hypno))[lights_on_epoch]

    sleep_onset = np.where(hypno!=0)[0][0] # first non-W epoch
    sleep_offset = np.where(hypno!=0)[0][-1] # last non-W epoch

    TST = (sum(hypno!=0)*epochlen)/60  # total time in sleep stages in minutes
    TBT = (lights_on_epoch-lights_off_epoch)*epochlen/60  # total time in bed
    TRT = len(hypno)*epochlen/60  # total recording time
    SE = np.round(TST/TBT, 2)  # sleep efficiency

    WASO = (sleep_offset-sleep_onset+1)*epochlen/60-TST
    min_S1 = sum(hypno==1)*epochlen/60
    min_S2 = sum(hypno==2)*epochlen/60
    min_S3 = sum(hypno==3)*epochlen/60
    min_REM = sum(hypno==4)*epochlen/60
    sum_NREM = min_S1 + min_S2 + min_S3

    perc_W = np.round(WASO/TBT * 100, 1)
    perc_S1 = np.round(min_S1/TST * 100, 1)
    perc_S2 = np.round(min_S2/TST * 100, 1)
    perc_S3 = np.round(min_S3/TST * 100, 1)
    perc_REM = np.round(min_REM/TST * 100, 1)

    # latency calculation
    for stage, name in enumerate(['lat_S1', 'lat_S2', 'lat_S3', 'lat_REM'], 1):
        if stage in hypno:
            locals()[name] = (np.argmax(hypno==stage)-lights_off_epoch)*epochlen/60
        else:
            # overwrite values with nan if the stage is not found at all
           warnings.warn(f'Stage for {name} not found, latency is NaN')
           locals()[name] = np.nan

    lights_off = lights_off_epoch*epochlen/60
    lights_on = lights_on_epoch*epochlen/60
    recording_length = len(hypno)*epochlen/60

    awakenings = 0
    stage_shifts =0
    mean_dur_awakenings = []

    # Skip wake at beginning and end
    for i, (stage, group) in enumerate(itertools.groupby(hypno)):
        group_list = list(group)
        if stage == 0 and i > 0 and i < len(list(itertools.groupby(hypno))) - 1:
            awakenings += 1
            mean_dur_awakenings.append(len(group_list) * epochlen / 60)
        elif i > 0:
            stage_shifts += 1

    mean_dur_awakenings = np.round(np.mean(mean_dur_awakenings), 1)

    FI = np.round((awakenings + stage_shifts) / TST, 2)
    SQI = np.round(SE * (1 - FI), 2)

    if SQI<0:
        SQI = np.nan

    # Number of Sleep Cycles
    # Define a sleep cycle as a transition from any NREM stage to REM
    sleep_cycles = 0
    in_rem = False
    for i in range(1, len(hypno)):
        if hypno[i] == 4 and not in_rem:
            sleep_cycles += 1
            in_rem = True
        elif hypno[i] != 4:
            in_rem = False

    summary = locals().copy()
    include = ['TST', 'TBT', 'TRT', 'SE', 'WASO', 'min_S1', 'min_S2', 'min_S3',
               'min_REM', 'sum_NREM', 'perc_W', 'perc_S1', 'perc_S2', 'perc_S3',
               'perc_REM', 'lat_S1', 'lat_S2', 'lat_S3', 'lat_REM', 'lights_off',
               'lights_on', 'recording_length', 'awakenings', 'stage_shifts',
               'mean_dur_awakenings', 'FI', 'SQI', 'sleep_cycles']
    summary = dict()

    for name in include:
        summary[name] = locals()[name]

    for name, value in summary.items():
        try:
            assert value>=0 or np.isnan(value), f'{name} has {value=}, should be positive or 0'
        except (TypeError, ValueError):
            warnings.warn(f'TypeError: {name} has type {type(value)}, unexpected.')

    if print_summary:
        pprint(summary)
    if sanity_check:
        hypno_check_summary(summary)
    return summary

def hypno_check_summary(summary, mode='warn'):
    """does a sanity check on hypnogram summary values"""

    def warn(msg):
        warnings.warn(msg)
    def exc(msg):
        raise ValueError(msg)

    if mode=='warn':
        action = warn
    elif mode=='raise':
        action = exc
    else:
        raise ValueError('mode must be "warn" or "raise"')

    s = AttrDict(summary)

    # Test NREM sum calculation
    if not s.sum_NREM == s.min_S1 + s.min_S2 + s.min_S3:
        action('Sum of S1+S2+S3 != NREM')

    # Test TRT calculation (assuming epochlen=30)
    if not s.TRT >=s.TBT:
        action(f'recording TRT must be longer than TBT: {s.TRT}<{s.TBT}?')


    total_percentage = s.perc_S1 + s.perc_S2 + s.perc_S3 + s.perc_REM
    if not np.isclose(total_percentage, 100, atol=0.1):
        action(f'{total_percentage}!=100 for sum of sleep stage percentages')

    for key, val in s.items():
        if val<0:
            action(f'negative value for {key}, not possible')

        # percentage cant be higher than 100%
        if key.startswith('perc_'):
            if s[key] > 100:
                action(f'{key} has > 100%: {val}')

        # latency cant be before lights off
        if key.startswith('lat_'):
            if s[key] < s.lights_off:
                f'{key} is before lights_off: {val} < {s.lights_off=}'

    if s.lights_off > s.lights_on:
        action(f'{s.lights_on=} before {s.lights_off}')

    if s.WASO*2<s.awakenings:
        action('{s.WASO=} minutes, but {s.awakenings=}?')

    if s.WASO<s.mean_dur_awakenings:
        action('{s.WASO=} minutes, but {s.mean_dur_awakenings=} minutes?')

    if s.TST > s.TBT or s.TST > s.TRT:
        action('something not right with calculation of TST, TBT or TST, check')


def infer_psg_file(hypno_file):
    """given a hypnogram file, try to find which data file matches to it"""
    folder, filename = os.path.split(hypno_file)
    file_noext = os.path.splitext(hypno_file)[0]

    possible_names = []
    for ext in ['.vhdr', '.fif', '.edf', '.bdf']:
        for suffix in ['', '_hypno', '_hypnogram']:
            if file_noext.endswith(suffix):
                possible_names += [file_noext.replace(suffix, '') + ext]
            else:
                possible_names += [file_noext + ext]

    for file in possible_names:
        if os.path.exists(os.path.join(folder, file)):
            return os.path.join(folder, file)
    warnings.warn(f'No Hypnogram found for {filename}, looked for: {possible_names}')
    return False

def infer_hypno_file(psg_file):
    """given a data file, try to find which hypnogram file matches to it"""
    folder, filename = os.path.split(psg_file)
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
            hypno_rep = [str(v) for v in np.repeat(hypno, 1)]
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


def hypno2wonambi(hypno, artefacts, dataset, winlen=10):
    """
    create annotations file

    :param hypno: hypnogram in 30 seconds base
    :param artefact: array with artefact markings
    :param dataset: a wonambi.Dataset type
    """
    import yasa
    from wonambi.attr import Annotations, create_empty_annotations

    # conv_dict = {0: 'Wake',
    #              1: 'NREM1',
    #              2: 'NREM2',
    #              3: 'NREM3',
    #              4: 'REM'}
    hypno_art = yasa.hypno_upsample_to_data(hypno, sf_hypno=1/30,
                                            data=artefacts, sf_data=1/10)
    with tempfile.NamedTemporaryFile(delete=True) as tmp_xls:
        create_empty_annotations(tmp_xls.name, dataset)
        annot = Annotations(tmp_xls.name)
        annot.add_rater('U-Sleep')

        while len((stages:=annot.rater.find('stages'))) != 0:
            for stage in stages:
                stages.remove(stage)

        annot.create_epochs(winlen)

        assert len(hypno_art)==len(artefacts)

        for i, (stage, art) in enumerate(zip(hypno_art, artefacts)):
            # name = conv_dict[int(stage)]
            name = str(stage)
            if art:
                annot.set_stage_for_epoch(i*winlen, 'Poor',
                                                     attr='quality',
                                                     save=False)
            else:
                annot.set_stage_for_epoch(i*winlen, name, save=False)

        annot.save()
    return annot

def get_var_from_comments(file, var, comments='#', typecast=int):
    """from a numpy txt file, extract variables that have been saved in
    the style
    '# var=X'
    """
    with open(file, 'r') as f:
        lines = f.read().split('\n')
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if line.startswith('#')]
        lines = [line for line in lines if f'{var}=' in line]
    if len(lines)==0:
        raise ValueError(f'variable "{var}" not found')
    elif len(lines)>1:
        raise ValueError(f'more than one variable "{var}" found')

    return typecast(lines[0].split('=')[1])


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


def make_random_hypnogram(
    n_epochs: int,
    average_cycle_length_minutes: float = 90.0,
    cycle_variation_fraction: float = 0.15,
    # This sets the “nominal” # of cycles we might try to fit in:
    # 8 hours / 90 minutes ~ 5 cycles, ±1
    base_n_cycles: int = None,
    n_awakenings: int = 5,
    seed: int = None
) -> np.ndarray:
    """
    Create a more realistic hypnogram with extra randomness:
      1) Vary # of cycles instead of a strict partitioning by total epochs.
      2) Randomize cycle lengths significantly around an average (±15% by default).
      3) Randomly reorder or skip some transitions (e.g. skip SWS in a late cycle).
      4) Randomly skip from SWS to REM without returning to N2, etc.
      5) Insert random awakenings plus random micro-arousals.

    Stages:
      0: Wake (W)
      1: N1
      2: N2
      3: SWS
      4: REM

    Each epoch is assumed to be 30 seconds.

    Parameters
    ----------
    n_epochs : int
        Total length of the hypnogram in 30-second epochs (e.g., 960 ≈ 8 hours).
    average_cycle_length_minutes : float
        Approximate duration of one full sleep cycle in minutes (~90 min typical).
    cycle_variation_fraction : float
        Fraction of average_cycle_length_minutes to use as ± random variation
        in each cycle’s length. (Default 0.15 → ±15%)
    base_n_cycles : int, optional
        Approximate # of full cycles to use. If None, automatically guesses
        from total epochs + average cycle length. (e.g., ~5 for 8 hours).
    n_awakenings : int
        Number of short awakenings (stage=0) to insert randomly.
    seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (n_epochs,) containing integer-coded sleep stages.
    """
    if seed is not None:
        np.random.seed(seed)

    # Helper function to clamp values
    def clamp(val, low, high):
        return max(low, min(high, val))

    # 1) Decide how many cycles we’re going to create in total
    #    If base_n_cycles is not specified, guess from the total time / cycle length
    if base_n_cycles is None:
        # total_time_minutes ~ n_epochs * 0.5
        total_time_minutes = n_epochs * 0.5
        guess = total_time_minutes / average_cycle_length_minutes  # e.g. ~8h / 90m = ~5.3
        # Round and add a random ±1
        base_n_cycles = int(round(guess + np.random.uniform(-1, 1)))
        base_n_cycles = clamp(base_n_cycles, 3, 8)  # typical range 3–8 cycles

    # 2) For each cycle, randomly determine cycle length (in epochs)
    #    Then build stages. Keep track until we fill up or exceed n_epochs.
    cycles_lengths = []
    for _ in range(base_n_cycles):
        # normal distribution around average_cycle_length_minutes
        stdev = average_cycle_length_minutes * cycle_variation_fraction
        c_minutes = np.random.normal(loc=average_cycle_length_minutes, scale=stdev)
        c_minutes = clamp(c_minutes, 60, 120)  # clamp from 1h–2h
        c_epochs = int(round(c_minutes * 2))   # convert minutes to 30s epochs
        cycles_lengths.append(c_epochs)

    # We’ll build the final stages in this list
    all_stages = []
    used_epochs = 0

    # Helper function to produce a single cycle’s stages
    def make_cycle(cycle_idx: int, cycle_length: int, total_cycles: int) -> list:
        """
        Build a single cycle with random durations for N1, N2, SWS, REM, possibly skipping transitions,
        or going SWS->REM directly. Returns a list of stages (integers).
        """
        stages_cycle = []

        # We define a typical flow: [N1, N2, SWS?, N2, REM, maybe N2 again],
        # but with random skipping or reordering.

        # N1 typically short: 5–10 min
        n1_epochs = np.random.randint(10, 21)  # 10–20 epochs → 5–10 min
        # SWS more in early cycles, less in later cycles
        # Sometimes we skip SWS in later cycles with small probability
        if cycle_idx < 2 or np.random.rand() > 0.2:  # 80% chance to have SWS
            if cycle_idx == 0:
                sws_epochs = np.random.randint(40, 61)  # 20–30 min
            elif cycle_idx == 1:
                sws_epochs = np.random.randint(30, 51)  # 15–25 min
            else:
                sws_epochs = np.random.randint(10, 31)  # 5–15 min
        else:
            sws_epochs = 0  # skip SWS

        # REM is short the first cycle, can get longer in later cycles
        if cycle_idx == 0:
            rem_epochs = np.random.randint(10, 16)   # ~5–8 min
        elif cycle_idx >= total_cycles - 1:
            rem_epochs = np.random.randint(35, 51)  # ~17–25 min on last cycle
        else:
            rem_epochs = np.random.randint(20, 31)  # ~10–15 min

        # If skipping SWS or from SWS→REM directly
        # 10-20% chance to jump from SWS directly to REM (skip N2 in between)
        skip_n2_between_sws_rem = False
        if sws_epochs > 0 and np.random.rand() < 0.2:
            skip_n2_between_sws_rem = True

        # Summation of the “fixed blocks”
        used_fixed = n1_epochs + sws_epochs + rem_epochs
        # cycle_length - used_fixed = leftover for N2 blocks around SWS/REM
        leftover_for_n2 = cycle_length - used_fixed

        # Minimum leftover for N2 if leftover is negative or very small
        leftover_for_n2 = max(leftover_for_n2, 10)

        # We’ll split leftover_for_n2 among up to 3 blocks of N2:
        # [N2 before SWS, (N2 between SWS/REM), N2 after REM]
        # but might skip the middle one if skip_n2_between_sws_rem is True
        r3 = np.random.rand(3)
        r3 /= r3.sum()  # random fractions summing to 1
        n2a = int(r3[0] * leftover_for_n2)
        n2b = int(r3[1] * leftover_for_n2) if not skip_n2_between_sws_rem else 0
        n2c = leftover_for_n2 - n2a - n2b

        # By now, we have something like:
        # N1 -> N2a -> SWS -> N2b -> REM -> N2c
        # but if SWS=0, we skip that block entirely

        # Build the cycle
        # 1) N1
        stages_cycle.extend([1] * n1_epochs)

        # 2) N2a
        stages_cycle.extend([2] * n2a)

        # 3) SWS
        if sws_epochs > 0:
            stages_cycle.extend([3] * sws_epochs)

        # 4) N2b (skip if skip_n2_between_sws_rem is True)
        if not skip_n2_between_sws_rem and sws_epochs > 0:
            stages_cycle.extend([2] * n2b)

        # 5) REM
        stages_cycle.extend([4] * rem_epochs)

        # 6) N2c
        stages_cycle.extend([2] * n2c)

        # Insert some random micro-arousals in the middle of this cycle
        # We'll do a small chance ~10% for each quarter of the cycle to insert a 1-2 epoch wake or N1
        parts = len(stages_cycle) // 4
        for k in range(1, 4):
            idx_start = k * parts
            if idx_start < len(stages_cycle) - 1 and np.random.rand() < 0.10:
                # flip 1–2 epochs to 0 or 1
                for offset in [0, 1]:
                    if idx_start+offset < len(stages_cycle) and np.random.rand() < 0.5:
                        stages_cycle[idx_start + offset] = np.random.choice([0, 1])

        return stages_cycle

    # Build each cycle in turn
    for i, c_len in enumerate(cycles_lengths):
        if used_epochs >= n_epochs:
            break
        # If adding another cycle would exceed, cap this cycle length
        if used_epochs + c_len > n_epochs:
            c_len = n_epochs - used_epochs
        # Create the cycle
        stages_this_cycle = make_cycle(
            cycle_idx=i,
            cycle_length=c_len,
            total_cycles=len(cycles_lengths)
        )
        all_stages.extend(stages_this_cycle)
        used_epochs += len(stages_this_cycle)

    # If we still have leftover epochs (rarely if cycle lengths sum < n_epochs),
    # just fill them with a final partial cycle
    if used_epochs < n_epochs:
        leftover = n_epochs - used_epochs
        # Treat it as last cycle
        partial_cycle = make_cycle(
            cycle_idx=len(cycles_lengths),
            cycle_length=leftover,
            total_cycles=len(cycles_lengths) + 1
        )
        all_stages.extend(partial_cycle[:leftover])
        used_epochs += leftover

    # 3) If somehow we exceeded (should not happen often), truncate
    if len(all_stages) > n_epochs:
        all_stages = all_stages[:n_epochs]

    # 4) Convert to array
    hypnogram = np.array(all_stages, dtype=int)

    # 5) Insert random awakenings (0) anywhere
    if n_awakenings > 0:
        awakening_epochs = np.random.choice(
            np.arange(1, n_epochs - 1),
            size=min(n_awakenings, n_epochs - 2),
            replace=False
        )
        for ep in awakening_epochs:
            hypnogram[ep] = 0
            # ~50% chance to extend the awakening
            if ep + 1 < n_epochs and np.random.rand() < 0.5:
                hypnogram[ep + 1] = 0

    return hypnogram


# No files created or modified during execution.


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
