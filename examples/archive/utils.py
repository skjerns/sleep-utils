# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 11:03:47 2022

utils for spindle analysis. Code mostly copied from other projects such as 
sleep_utils etc etc.

@author: Simon
"""

import sys
import os
import mne
import subprocess
import warnings

from pyedflib import highlevel
import numpy as np
from io import StringIO
from sleep_utils.sigproc import resample
from tkinter.filedialog import askopenfilenames, askdirectory
from tkinter import simpledialog
from tkinter import  Tk
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter, median_filter
from joblib import memory
from xml.etree.ElementTree import Element, SubElement


mem = memory.Memory('./tmp-cache/')


stages_dict = {'WAKE':0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4, **{i:i for i in range(5, 10)}}
stages_dict = {v: k for k, v in stages_dict.items()}


def install(package, options):
    if isinstance(package, str):
        package = [package]
    if isinstance(options, str):
        options = [options]    
    with subprocess.Popen([sys.executable, "-m", "pip", "install", *package, *options], stdout=subprocess.PIPE, bufsize=0) as p:
        char = p.stdout.read(1)
        while char != b'':
            print(char.decode('UTF-8'), end='', flush=True)
            char = p.stdout.read(1)
    if p.returncode: 
        raise Exception(f'\t!!! Could not install {package}\n')


def choose_files(default_dir=None, exts='txt', title='Choose file(s)'):
    """
    Open a file chooser dialoge with tkinter.
    
    :param default_dir: Where to open the dir, if set to None, will start at wdir
    :param exts: A string or list of strings with extensions etc: 'txt' or ['txt','csv']
    :returns: the chosen file
    """
    root = Tk()
    root.iconify()
    root.update()
    if isinstance(exts, str): exts = [exts]
    files = askopenfilenames(initialdir=None,
                           parent=root,
                           title = title,
                           filetypes =(*[("File", "*.{}".format(ext)) for ext in exts],
                                       ("All Files","*.*")))
    root.update()
    root.destroy()

    return files
    
@mem.cache()
def load_edf(edf_file, channels, references, crop_to_hypno=True):
    """convenience function to load an EDF+ with references without MNE.
    Workaround for https://github.com/mne-tools/mne-python/issues/10635"""
    
    if not isinstance(channels, list):
        channels = [channels]
    if not isinstance(references, list):
        references = [references]
        
    sigs, sigheads, header = highlevel.read_edf(edf_file, 
                                                ch_names=channels+references)
    sigs = [sig * 1e-6 for sig in sigs]
    
    assert [shead['label'] in channels for shead in sigheads[:len(channels)]]
    assert [shead['label'] in references for shead in sigheads[len(channels):]]

    sfreqs = [s['sample_rate'] for s in sigheads]
    labels = [s['label'] for s in sigheads]
    
    if len(set(sfreqs))>1:
        sigs = [resample(x, sfreqs[i], min(sfreqs)) for i, x in enumerate(sigs)]

    if crop_to_hypno:
        sigs = [sig[:int(len(sig)-len(sig)%(30*min(sfreqs)))] for sig in sigs]

    info = mne.create_info(labels, min(sfreqs), ch_types='eeg')
    raw = mne.io.RawArray(sigs, info)
    raw._filenames = [edf_file.replace('/', '\\')]
    
    raw, _ = mne.set_eeg_reference(raw, ref_channels=references)
    raw.pick(channels)
    
    return raw

    
def infer_hypno_file(filename):
    folder, filename = os.path.split(filename)
    possible_names = [filename + '.txt']
    possible_names += [filename + '.csv']
    possible_names += [os.path.splitext(filename)[0] + '.txt']
    possible_names += [os.path.splitext(filename)[0] + '.csv']


    for file in possible_names:  
        if os.path.exists(os.path.join(folder, file)):
            return os.path.join(folder, file)
    warnings.warn(f'No Hypnogram found for {filename}, looked for: {possible_names}')
    return False


def get_var_from_comments(file, var, comments='#', dtype=int):
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
    
    return int(lines[0].split('=')[1])


def list_files(folder, ext='edf'):
    """small helper function that lists a folders contents"""
    files = [os.path.join(folder, f) for f in os.listdir(folder)]
    files = [file for file in files if file.endswith(ext)]
    return files


def get_subj_cond(filename):
    """
    split filename (e.g. AF4_EX_(1).edf) into subject (i.e. AF4) and
    condition (i.e. EX)
    """
    subj = os.path.basename(filename)[:3]
    if 'REST' in filename.upper():
        cond = 'REST'
    elif 'EX' in filename.upper():
        cond = 'EX'
    else:
        raise ValueError('Neithe EX nor REST found in {filename}')
    return subj, cond



