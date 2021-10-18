# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 11:22:20 2018

@author: skjerns

This file contains all functions that deal with IO

"""
import os
import numpy as np
import time
import mne
import ospath
from io import StringIO
import warnings
from datetime import datetime 


def sleep(seconds):
    if seconds > 1:
        for s in range(seconds):
            time.sleep(1)
    else:
        time.sleep(seconds)
        
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        

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

    
def read_hypno(hypno_file, epochlen = 30, epochlen_infile=None, mode='auto', 
               exp_seconds=None, conf_dict=None):
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
        conv_dict = {'W':0, 'WAKE':0, 'N1': 1, 'N2': 2, 'N3': 3, 'R':4, 'REM': 4, 'ART': 5,
                     -1:8, **{i:i for i in range(9)}, **{f'{i}':i for i in range(9)}}
    
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
            print('[INFO] Assuming csv annotations with one entry per {} seconds'.format(epochlen_infile))

        elif epochlen_infile is None: 
            if len(lines) < 2400: # we assume no recording is longer than 20 hours
                epochlen_infile = 30
                print('[INFO] Assuming csv annotations are per epoch')
            else:
                epochlen_infile = 1
                print('[INFO] Assuming csv annotations are per second')
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
        print('[WARNING] hypnogram loading failed, len == 0')
        
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
    assert not ospath.exists(filename) or overwrite, 'File already exists, no overwrite'
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
    assert not ospath.exists(filename) or overwrite, 'File already exists, no overwrite'
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
    if not filename.endswith('.hypno'):
        warnings.warn('Filename ends in ".{}", recommended to end in .hypno'.format(
                       os.path.splitext(filename)[1]))
    assert seconds_per_annotation>=1
    if mode=='time':
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
    assert not ospath.exists(hypno_file_out)
    hypno = read_hypno(hypno_file_in, **kwargs)
    return write_hypno(hypno, hypno_file_out)
    


def _time_format(seconds):
    """ returns a string with a fitting string"""
    return '{:02d}:{:02d} hours'.format(seconds//3600, seconds%3600//60)


    

def write_mne_edf(mne_raw, fname, picks=None, tmin=0, tmax=None, overwrite=False):
    """
    Saves the raw content of an MNE.io.Raw and its subclasses to
    a file using the EDF+ filetype
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
    if not issubclass(type(mne_raw), mne.io.BaseRaw):
        raise TypeError('Must be mne.io.Raw type')
    if not overwrite and os.path.exists(fname):
        raise OSError('File already exists. No overwrite.')
    # static settings
    sfreq = mne_raw.info['sfreq']
    date = datetime.now().strftime( '%d %b %Y %H:%M:%S')
    first_sample = int(sfreq*tmin)
    last_sample  = int(sfreq*tmax) if tmax is not None else None

    if 'STI 014' in mne_raw.ch_names:
        mne_raw.drop_channels(['STI 014'])
    # convert data
    channels = mne_raw.get_data(picks, 
                                start = first_sample,
                                stop  = last_sample)
    
    # convert to microvolts to scale up precision
    channels *= 1e6
    
    # set conversion parameters
    dmin, dmax = [-32768,  32767]
    pmin, pmax = [channels.min(), channels.max()]
    # create channel from this   

    header = make_header('mne-gist-save-edf-skjerns', startdate=date)
    signal_headers = []
    for i in range(len(channels)):
        signal_header = make_signal_header(label=mne_raw.ch_names[i],
                                           sample_rate=sfreq,
                                           physical_min=pmin,
                                           physical_max=pmax,
                                           digital_min=dmin,
                                           digital_max=dmax)
        signal_headers.append(signal_header)

    write_pyedf(fname, channels, signal_headers, header)
    return True


            
        
