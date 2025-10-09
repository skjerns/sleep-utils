# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 17:00:08 2022

@author: Simon
"""

import os
import importlib
try:
    import utils
except ModuleNotFoundError:
    raise ModuleNotFoundError('Could not find utils script - please check you are' 
                              'running in the same folder as the spindle repository,'
                              ' and that utils.py is located there')

print('Loading libraries...')
utils.install('sleep-utils', ['--upgrade', '--no-deps'])

libraries = ['llvmlite', 'numpy', 'easygui', 'tqdm', 'pyedflib', 'pandas']
for lib in libraries:
    try:
        print(f'Loading {lib}...')
        importlib.import_module(lib)
    except ModuleNotFoundError:
        print(f'Library `{lib}` not found, attempting to install')
        utils.install(lib)
        
try:
    import pandas as pd
    import warnings
    import numpy as np
    import easygui
    from tqdm import tqdm
    from pyedflib import highlevel
    import matplotlib.pyplot as plt
    import sleep_utils
    from sleep_utils.sigproc import resample
except ModuleNotFoundError as e:
    
    raise ModuleNotFoundError(f'Could not find library {e.name}\nPlease install '
                              f'via the command `pip install {e.name}`')


#%%

if __name__=='__main__':
    edf_files = utils.choose_files(exts='edf', title='Choose files for analysis')
    
    ch_ignore = ['II','EKG II', 'EMG1', 'Akku', 'Akku Stan', 'Lage', 'Licht', 
                 'Aktivitaet', 'SpO2', 'Pulse',  'Pleth', 'Flow&Snor', 'RIP Abdom',
                 'RIP Thora', 'Summe RIP', 'RIP', 'Stan', 'Abdom', 'Thora']
    
    participants = set([file.rpartition('/')[-1][:3] for file in edf_files])
    print(f'Identified {len(participants)} participants: {participants}')
    
    all_chs = set()
    common_chs = set()
    
    for edf_file in tqdm(edf_files, desc='Scanning files'):
        # first check the channels that are available
        header = highlevel.read_edf_header(edf_file)
        duration = header['Duration']
        chs = [ch for ch in header['channels'] if not ch in ch_ignore]
        if len(common_chs)==0: 
            common_chs = set(chs)
        common_chs.intersection(chs)
        all_chs = all_chs.union(chs)
        
    common_chs = sorted(common_chs)
    
    preselect_ch = common_chs.index('Pz') if 'Pz' in common_chs else 0
    preselect_ref = [common_chs.index('M1')+1, common_chs.index('M2')+1] if 'M1' in common_chs else 0
    
    ch = easygui.choicebox('Please select main channel for creating the power analysis.\nHere is a list of channels that is available in all recordings.',  
                           choices=common_chs, preselect=preselect_ch)
    ref = easygui.multchoicebox(f'Please select one or more REFERENCE for {ch}', preselect=preselect_ref,
                            choices=['no referencing'] + common_chs)
    
    assert ch, 'no channel selected'
    assert ref, 'no reference selected'
    
    
    #%%
    fig = plt.figure(figsize=[12,6])
    axs = fig.subplots(1, 3); axs=axs.flatten()
            
    tqdm_loop = tqdm(total=len(edf_files), desc='creating spectrograms')

    power_N2 = []
    power_N3 = []
    power_NREM = []

    for i, subj in enumerate(tqdm(participants)):
       
        edf_subj = [file for file in edf_files if subj in file]
            
        conditions = ['EX', 'REST']
        
        for ax in axs:
            ax.clear()

        res_N2 = {}
        res_N3 = {}
        res_NREM = {}

        for cond in conditions:
            edf_file = [file for file in edf_subj if cond.lower() in file.lower()][0]
            filedesc = os.path.basename(edf_file)[:-4]
            # we load the file into memory using the function `mne.io.read_raw_edf`
            tqdm_loop.set_description('Loading file')
           
            sig, sigheads, header = highlevel.read_edf(edf_file, ch_names = [ch] + ref)
            
            sfreqs = [s['sample_rate'] for s in sigheads]
            if 'no referencing' in ref:
                # set the reference
                data = sig[0]
            else:
                refdata = np.atleast_2d(sig[1:])
                if len(set(sfreqs))>1:
                    tqdm_loop.set_description('Resampling')
    
                    # print(f'Not all sampling frequencies are the same for {ch} and {ref}: {sfreqs}')
                    refdata = [resample(x, sfreqs[i+1], sfreqs[0]) for i, x in enumerate(refdata)]
                data = sig[0] - np.mean(refdata, 0)
                
            hypno_file = utils.infer_hypno_file(edf_file)
            if hypno_file:
                hypno = sleep_utils.read_hypno(hypno_file, verbose=False)
            else:
                print( f'No hypnogram found for {edf_file}, make sure it\'s in the same folder')
            

            
            hypno_len = int(len(hypno)*30*sfreqs[0])
            if abs(len(data)-hypno_len) > 30*sfreqs[0]:
                warnings.warn('Data is more than 30 seconds longer than hypnogram!')
                
            data = data[:hypno_len] # trim data to hypnogram length
            hypno_10sec = np.repeat(hypno, 3)
            win_len = int(10 * sfreqs[0])
            
            power = {x:[] for x in set(hypno)}
            
            for samp, stage in zip(range(0, len(data), win_len), hypno_10sec):
                win = (np.abs(np.fft.rfft(data[samp:samp+win_len]))**2)[:-1]
                power[stage].append(win)
            
            freqs = np.fft.fftfreq(win_len, 1/sfreqs[0])[:len(win)]
            
            
            mean_N2 = np.mean(power[2], 0)
            mean_N3 = np.mean(power[3], 0)
            mean_NREM = np.mean(np.vstack([power[2], power[3]]), 0)
            
            res_N2[cond] = mean_N2
            res_N3[cond] = mean_N3
            res_NREM[cond] = mean_NREM

            
            axs[0].plot(freqs, np.log10(mean_N2)*20)
            axs[1].plot(freqs, np.log10(mean_N3)*20)
            axs[2].plot(freqs, np.log10(mean_NREM)*20)
            
        power_N2.append(pd.Series(res_N2, name=subj))
        power_N3.append(pd.Series(res_N3, name=subj))
        power_NREM.append(pd.Series(res_NREM, name=subj))

        for ax,desc in zip(axs, ('N2', 'N3', 'N2+N3')):
            ax.legend(conditions)
            ax.set_xlabel('frequency')
            ax.set_ylabel('power (dB)')
            ax.set_title(desc)
            fig.suptitle(f'Power spectrum for different conditions for {subj}')
        
        png_file = edf_file + f'.power_{ch}.png'
        fig.savefig(png_file)
        tqdm_loop.update()

    #%% calculate statistics
    
    N2_EX = df_N2['EX'].to_numpy()
    N2_REST = df_N2['REST']

    N3_EX = df_N3['EX']
