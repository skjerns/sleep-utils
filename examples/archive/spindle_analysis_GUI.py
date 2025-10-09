# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 11:06:38 2022

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
libraries = ['llvmlite', 'pandas', 'numpy', 'mne', 'yasa', 'easygui', 'tqdm', 'pyedflib', 'joblib']
for lib in libraries:
    try:
        print(f'Loading {lib}...')
        importlib.import_module(lib)
    except ModuleNotFoundError:
        print(f'Library `{lib}` not found, attempting to install')
        utils.install(lib)
        
try:
    import os
    import mne
    import yasa
    import easygui
    import pandas as pd
    import numpy as np
    from joblib import Parallel, delayed, cpu_count
    from tqdm import tqdm
    from pyedflib import highlevel
except ModuleNotFoundError as e:
    
    raise ModuleNotFoundError(f'Could not find library {e.name}\nPlease install '
                              f'via the command `pip install {e.name}`')



if __name__=='__main__':
    print('Please select EDF files for loading..')
    #%% choose files to run analysis on
    edf_files = utils.choose_files(exts='edf')
    
    # edf_files = ('Z:/Exercise_Sleep_Project/EDF Export EEG/AA3_EX_AA3_EX_(1).edf',
    #   'Z:/Exercise_Sleep_Project/EDF Export EEG/AA3_REST_AA3_REST_(1).edf')
    
    #%% check all files and scan channels
    
    # ignore these channels in the selection
    ch_ignore = ['II','EKG II', 'EMG1', 'Akku', 'Akku Stan', 'Lage', 'Licht', 
                 'Aktivitaet', 'SpO2', 'Pulse',  'Pleth', 'Flow&Snor', 'RIP Abdom',
                 'RIP Thora', 'Summe RIP', 'RIP', 'Stan', 'Abdom', 'Thora']
    all_chs = set()
    common_chs = set()
    common_stages = []
    
    for edf_file in tqdm(edf_files, desc='Scanning files'):
        # first check the channels that are available
        header = highlevel.read_edf_header(edf_file)
        duration = header['Duration']
        chs = [ch for ch in header['channels'] if not ch in ch_ignore]
        if len(common_chs)==0: 
            common_chs = set(chs)
        common_chs.intersection(chs)
        all_chs = all_chs.union(chs)
        
        # now check that there is a hypnogram available
        hypno_file = utils.infer_hypno_file(edf_file)
        assert hypno_file, f'No hypnogram found for {edf_file}, make sure it\'s in the same folder'
        hypno = utils.read_hypno(hypno_file, verbose=False, exp_seconds=duration)
        common_stages.extend(hypno)
        assert (len(hypno)-(duration/30))<2, \
            f'hypnogram does not match length of file: hypno {len(hypno)} epochs, file {(duration/30)} epochs'
        
    common_chs = sorted(common_chs)
    #%% select channel to run analysis on
    
    preselect_ch = common_chs.index('C4') if 'C4' in common_chs else 0
    preselect_ref = common_chs.index('M1') if 'C4' in common_chs else 0
    
    ch = easygui.choicebox(f'Please select main channel for running the spindle analysis.\nHere is a list of channels that is available in all recordings.',  
                           choices=common_chs, preselect=preselect_ch)
    ref = easygui.choicebox(f'Please select REFERENCE for {ch}', preselect=preselect_ref,
                            choices=['no referencing'] + common_chs)
    
    assert ch, 'no channel selected'
    assert ref, 'no reference selected'
    
    #%% select stages to run analysis on
    stages = set(common_stages)
    stages = [f'{s} - {utils.stages_dict[s]}' for s in stages]
    
    stages_sel = easygui.multchoicebox(f'Which stages do you want to limit the analysis to?',
                               choices=stages, preselect=[2, 3])
    stages_sel = [int(x[0]) for x in stages_sel]
    
    report_mode = easygui.buttonbox(f'Do you want the report on the selected stages ({stages_sel}) together or each stage individually?',
                      choices=['together', 'individually'])
    
    params_in = {'Spindle lower frequency limit':12,
                 'Spindle upper frequency limit':15,
                 'Minimum spindle duration in ms': 500,
                 'Maximum spindle duration in ms': 2000,
                 'Minimum spindle distance in ms': 500}
    
    
    params_out = easygui.multenterbox(msg='Please select the parameters for the spindle analysis',
                         fields=list(params_in.keys()), values=list(params_in.values()))
    params_out = [x.replace(',', '.') for x in params_out] # silly Germans, using comma as decimal points
    freq_sp = (float(params_out[0]), float(params_out[1]))
    duration = (int(params_out[2])/1000, int(params_out[3])/1000)
    min_distance = int(params_out[3])
    
    run_name = easygui.enterbox('(optional) Give a name to your analysis (will be pre-pended to the results filename)\n'
                                'Leave empty if you don\'t want to name your analysis')
    if run_name: run_name += '_'
    
    method = 'yasa'


#%% run spindle analysis

tqdm_loop = tqdm(total=len(edf_files))
def set_desc(desc):
    tqdm_loop.set_description(f'{desc} - {os.path.basename(edf_file)}')


all_summary = pd.DataFrame()

for edf_file in edf_files:
    subj = os.path.basename(edf_file)    

    summary_csv = os.path.join(os.path.dirname(edf_file), f'_summary_{run_name}n{len(edf_files)}_{method}.csv')
    spindles_csv = f'{run_name}{subj}_spindles_{method}.csv'
    
    set_desc('Loading file')
    ch_ignore = all_chs.difference([ch, ref])
    
    raw = mne.io.read_raw_edf(edf_file, exclude=ch_ignore, preload=True, verbose='WARNING') # we load the file into memory using the function `mne.io.read_raw_edf`
    if ref!='no referencing':
        raw = mne.set_bipolar_reference(raw, anode=[ch], cathode=[ref], verbose=False)
        ch_pick = f'{ch}-{ref}'
    else:
        ch_pick = ch
    raw.pick_channels([ch_pick])
    
    hypno_file = utils.infer_hypno_file(edf_file)
    hypno = utils.read_hypno(hypno_file, verbose=False)
    hypno_resampled = yasa.hypno_upsample_to_data(hypno, sf_hypno=1/30, data=raw, verbose='ERROR')

    set_desc('Detecing spindles')
    raw._data += 1e-16 # small offset to prevent division by 0
    spindles = yasa.spindles_detect(raw, hypno=hypno_resampled, include=stages_sel)    

    spindles_subj = spindles.summary()
    spindles_subj.to_csv(spindles_csv)
    
    spindles_subj.drop(['Start', 'Peak', 'End'], inplace=True, axis=1)
    

    if report_mode=='together':
        summary = spindles_subj.mean(0)
        samples_stages = sum([(hypno_resampled==s).sum() for s in stages_sel])
        stages_minutes = samples_stages/60/raw.info['sfreq']
        summary['Density'] = len(spindles_subj)/stages_minutes
        summary['Stage'] = ' & '.join([str(x) for x in stages_sel])
        summary.name = subj
        all_summary = pd.concat([all_summary, summary])
    elif report_mode=='individually':
        for stage in stages_sel:
            if not stage in spindles_subj['Stage']: continue
            spindles_stage = spindles_subj[spindles_subj['Stage']==stage]
            summary = spindles_stage.mean(0)
            samples_stages = np.nansum(hypno_resampled==stage)
            stages_minutes = samples_stages/60/raw.info['sfreq']
            summary['Density'] = len(spindles_stage)/stages_minutes
            summary['Stage'] = stage
            summary.name = subj
            all_summary = pd.concat([all_summary, summary])
    else:
        raise ValueError(f'mode not known: {report_mode}')
        
    tqdm_loop.update()    
    all_summary['Stage'] = all_summary['Stage'].astype(str)
    all_summary = all_summary.sort_values('Stage')
    all_summary.to_csv(summary_csv)



try:
    import pandasgui
    pandasgui.show(all_summary)
except:
    print('INFO: pandasgui is not installed, cannot show results. To visualize the results, install via `pip install pandasgui`')