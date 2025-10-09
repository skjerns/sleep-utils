# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 12:57:00 2020

RR_windows contains RR intervals in milliseconds!

@author: Simon Kern
"""
import logging as log
import numpy as np
import hrvanalysis
import entropy # pip install git+https://github.com/raphaelvallat/entropy.git

# ### caching dir to prevent recomputation of reduntant functions
# if hasattr(cfg, 'folder_cache'):
#     print(f'caching enabled in features.py to {cfg.folder_cache}')
#     memory = Memory(cfg.folder_cache, verbose=0)
#     # memory = Memory(None, verbose=0)
# else:
#     print('caching disabled in features.py')
#     memory = Memory(None, verbose=99)
# ###################################################

def resp_freq(thorax_windows, sfreq, **kwargs):
    """
    calculate the respiratory frequency distribution based on the 
    thorax windows.
    """
    feat = []
    for windows in thorax_windows:
        pass
    return np.array(feat)

def rrHRV(RR_windows, **kwargs):
    """
    A new geometric measure for HRV is introduced.
    It is based on relative RR intervals, the difference
    of consecutive RR intervals weighted by their mean.
    """
    def euclidean(p, q):
        return np.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)

    feat = []
    for wRR in RR_windows:
        if len(wRR)<2:
            rrHRV = np.nan
        else:
            wRR = np.array(wRR)
            rr_i = (2*(wRR[1:]-wRR[:-1])) /  (wRR[1:]+wRR[:-1])
            return_map = np.vstack([rr_i[:-1], rr_i[1:]]) # (rr[i], rr[i+1])
            center = np.mean(return_map[:, rr_i[:-1]<=0.2], 1) # center of map with all |rr[i]| < 20%
            distances = euclidean(return_map, center)
            rrHRV = np.median(distances)
        feat.append(rrHRV)

    return np.array(feat)


def dummy(RR_windows, **kwargs):
    """
    each function here should be named exactly as the feature 
    name in config.features_mapping.
    Like this the feature extraction can be done automatically.
    The function can accept the following parameters:
        
    RR_windows: a list of windows with RR intervals denoted in SECONDS 
    
    IMPORTANT: Note that RRs are in SECONDS not MILLISECONDS
               and need to be converted if necessary inside the feature funcs
    
    all functions should accept **kwargs that are ignored.
    """
    pass
    
def identity(RR_windows, **kwargs):
    return RR_windows

def lengths(RR_windows, **kwargs):
    """
    return how many seconds each window spans
    mainly interesting for debugging purposes
    """
    lengths = [np.sum(wRR) for wRR in RR_windows]
    return lengths

# 1  
def mean_HR(RR_windows, **kwargs):
    feat = []
    for wRR in RR_windows:
        if len(wRR)<2:
            HR = np.nan
        else:
            HR = len(wRR)/np.sum(wRR)*60
        feat.append(HR)
    return np.array(feat)

# 2
def mean_RR(RR_windows, **kwargs):
    feat = []
    for wRR in RR_windows:
        if len(wRR)<1:
            mRR = np.nan
        else:
            mRR = np.mean(wRR)
        feat.append(mRR)
    return np.array(feat)

#3
def SDNN(RR_windows, **kwargs):
    feat = []
    for wRR in RR_windows:
        if len(wRR)<1:
            SDNN = np.nan
        else:
            SDNN = np.std(wRR)

        feat.append(SDNN)
    return np.array(feat)

# 4
def RMSSD(RR_windows, **kwargs):
    feat = []
    for wRR in RR_windows:
        if len(wRR)<2:
            RNSSD = np.nan
        else:
            diffRR = np.diff(wRR)
            RNSSD = np.sqrt(np.mean(diffRR ** 2))
        feat.append(RNSSD)
    return np.array(feat)

#5
def RR_range(RR_windows, **kwargs):
    feat = []
    for wRR in RR_windows:
        if len(wRR)<2:
            value = np.nan
        else:
            value = np.ptp(wRR)
        feat.append(value)
    return feat

# 6
def pNN50(RR_windows, **kwargs):
    return pNNxx(RR_windows, XX=50)


# 8
def SDSD(RR_windows, **kwargs):
    feat = []
    for wRR in RR_windows:
        if len(wRR)<2:
            SDSD = np.nan
        else:
            diffRR = np.diff(wRR)
            SDSD = np.std(diffRR)
        feat.append(SDSD)
    return np.array(feat)

def ULF_power(RR, **kwargs):
    ulf_band = (0, 0.0033)
    feat = hrvanalysis.get_frequency_domain_features((RR*1000).astype(int), vlf_band=ulf_band,
                                                            sampling_frequency=10, interpolation_method='cubic')
    feat = feat['vlf']
    return np.array(feat)

def VLF_power(RR_windows, **kwargs):
    feat = get_frequency_domain_features(RR_windows)['vlf']
    return np.array(feat)


# 10
def LF_power(RR_windows, **kwargs):
    feat = get_frequency_domain_features(RR_windows)['lf']
    return np.array(feat)

# 11
def HF_power(RR_windows, **kwargs):
    feat = get_frequency_domain_features(RR_windows)['hf']
    return np.array(feat)




# 12
def LF_HF(RR_windows, **kwargs):
    feat = get_frequency_domain_features(RR_windows)['lf_hf_ratio']
    return np.array(feat)
   
def pNNxx(RR_windows, xx=50, **kwargs):
    """
    Calculate the pNN index for a given millisecond interval difference
    pNN50 is the percentage of successive beats that differ more than 50ms
    """
    feat = []
    for wRR in RR_windows:
        if len(wRR)<2:
            pNN50 = np.nan
        else:
            diffRR = np.diff(wRR)
            pNN50 = ((diffRR>(xx/1000)).sum()/len(diffRR))*100
        feat.append(pNN50)
    return np.array(feat)


def SD1(RR_windows, **kwargs):
    feat = get_poincare_plot_features(RR_windows)['sd1']
    return feat

def SD2(RR_windows, **kwargs):
    feat = get_poincare_plot_features(RR_windows)['sd2']
    return feat

def SD2_SD1(RR_windows, **kwargs):
    feat = get_poincare_plot_features(RR_windows)['ratio_sd2_sd1']
    return feat

# Not Implemented
# def TINN(RR_windows, **kwargs):
#     feat = get_geometrical_features(RR_windows)['tinn']
#     return feat

def triangular_index(RR_windows, **kwargs):
    feat = get_geometrical_features(RR_windows)['triangular_index']
    return feat

# def TINN(RR_windows):
#     feat = []
#     for wRR in RR_windows:
#         if len(wRR)<2:
#             value = np.nan
#         else:
#             value = pyhrv.time_domain.time_domain(wRR, plot=False, show=False)
#         feat.append(value)
#     return np.array(feat)

def SNSindex(RR_windows, **kwargs):
    feat = get_csi_cvi_features(RR_windows)['csi']
    return feat

def PNSindex(RR_windows, **kwargs):
    feat = get_csi_cvi_features(RR_windows)['cvi']
    return feat

def modified_csi(RR_windows, **kwargs):
    feat = get_csi_cvi_features(RR_windows)['Modified_csi']
    return feat




################### Entropy features
# 67
def SampEn(RR_windows, **kwargs):
    feat = []
    for wRR in RR_windows:
        if len(wRR)<8:
            value = np.nan
        else:
            # value = nolds.sampen(wRR, emb_dim=1)
            value = entropy.sample_entropy(wRR, order=2, metric='chebyshev')
        feat.append(value)
    return np.array(feat)


def PermEn(RR_windows, **kwargs):
    feat = []
    for wRR in RR_windows:
        try:
            value = entropy.perm_entropy(wRR, order=3, normalize=True)
        except:
            value = np.nan
        feat.append(value)
    return feat


def SVDEn(RR_windows, **kwargs):
    # Singular value decomposition entropy
    feat = []
    for wRR in RR_windows:
        try:
            value = entropy.svd_entropy(wRR, order=3, delay=1, normalize=True)
        except:
            value = np.nan
        feat.append(value)
    return feat

def ApEn(RR_windows, **kwargs):
    feat = []
    for wRR in RR_windows:
        try:
            value = entropy.app_entropy(wRR, order=2, metric='chebyshev')
        except:
            value = np.nan
        feat.append(value)
    return feat



################### Fractal features


def PetrosianFract(RR_windows, **kwargs):
    feat = []
    for wRR in RR_windows:
        try:
            value = entropy.petrosian_fd(wRR)
        except:
            value = np.nan
        feat.append(value)
    return feat

def KatzFract(RR_windows, **kwargs):
    feat = []
    for wRR in RR_windows:
        try:
            value = entropy.katz_fd(wRR)
        except:
            value = np.nan
        feat.append(value)
    return feat

def HiguchiFract(RR_windows, **kwargs):
    feat = []
    for wRR in RR_windows:
        try:
            value = entropy.higuchi_fd(wRR, kmax=5)
        except:
            value = np.nan
        feat.append(value)
    return feat

def detrend_fluctuation(RR_windows, **kwargs):
    feat = []
    for wRR in RR_windows:
        try:
            value = entropy.detrended_fluctuation(wRR)
        except:
            value = np.nan
        feat.append(value)
    return feat


##############################################################################
##############################################################################
### Helper functions

def get_csi_cvi_features(RR_windows):
    assert isinstance(RR_windows, (list, np.ndarray))
    if isinstance(RR_windows, np.ndarray): 
        assert RR_windows.ndim==2, 'Must be 2D'
        
    if any([np.any(wRR>1000) for wRR in RR_windows if len(wRR)>0]):
        log.warn('Values seem to be in ms instead of seconds! Algorithm migh fail.')

    feats = { x:[] for x in ['csi', 'cvi', 'Modified_csi']}
    for wRR in RR_windows:
        if len(wRR)<2:
            for key, val in feats.items(): feats[key].append(np.nan)
        else:
            mRR = hrvanalysis.get_csi_cvi_features((wRR*1000).astype(int))
            for key, val in mRR.items(): feats[key].append(val)
    return feats



def get_geometrical_features(RR_windows):
    assert isinstance(RR_windows, (list, np.ndarray))
    if isinstance(RR_windows, np.ndarray): 
        assert RR_windows.ndim==2, 'Must be 2D'
        
    if any([np.any(wRR>1000) for wRR in RR_windows if len(wRR)>0]):
        log.warn('Values seem to be in ms instead of seconds! Algorithm migh fail.')

    feats = { x:[] for x in ['tinn', 'triangular_index']}
    for wRR in RR_windows:
        if len(wRR)<2:
            for key, val in feats.items(): feats[key].append(np.nan)
        else:
            mRR = hrvanalysis.get_geometrical_features((wRR*1000).astype(int))
            for key, val in mRR.items(): feats[key].append(val)
    return feats

def get_poincare_plot_features(RR_windows):
    assert isinstance(RR_windows, (list, np.ndarray))
    if isinstance(RR_windows, np.ndarray): 
        assert RR_windows.ndim==2, 'Must be 2D'
        
    if any([np.any(wRR>1000) for wRR in RR_windows if len(wRR)>0]):
        log.warn('Values seem to be in ms instead of seconds! Algorithm migh fail.')

    feats = { x:[] for x in ['sd1', 'sd2', 'ratio_sd2_sd1']}
    for wRR in RR_windows:
        if len(wRR)<2:
            for key, val in feats.items(): feats[key].append(np.nan)
        else:
            mRR = hrvanalysis.get_poincare_plot_features((wRR*1000).astype(int))
            for key, val in mRR.items(): feats[key].append(val)
    return feats


def get_frequency_domain_features(RR_windows):
    """
    Calculate frequency domain features of this RR.
    This function is being cached as it computes a bunch of features
    at the same time 
    
    returns
    {    'lf': 0.0,
         'hf': 0.0,
         'lf_hf_ratio': nan,
         'lfnu': nan,
         'hfnu': nan,
         'total_power': 0.0,
         'vlf': 0.0}
    """
    assert isinstance(RR_windows, (list, np.ndarray))
    if isinstance(RR_windows, np.ndarray): 
        assert RR_windows.ndim==2, 'Must be 2D'
        
    if any([np.any(wRR>1000) for wRR in RR_windows if len(wRR)>0]):
        log.warn('Values seem to be in ms instead of seconds! Algorithm migh fail.')
        
    feats = { x:[] for x in ['lf', 'hf', 'lf_hf_ratio', 'lfnu', 'hfnu', 'total_power', 'vlf']}
    for wRR in RR_windows:
        if len(wRR)<5: # fewer than 5 beats in this window? unlikely
            for key, val in feats.items(): feats[key].append(np.nan)
        else:
            mRR = hrvanalysis.get_frequency_domain_features((wRR*1000).astype(int),
                                                            sampling_frequency=10, interpolation_method='cubic')
            for key, val in mRR.items(): feats[key].append(val)
    return feats



### other functions

def _window_view(a, window, step = None, axis = None, readonly = True):
        """
        Create a windowed view over `n`-dimensional input that uses an 
        `m`-dimensional window, with `m <= n`

        Parameters
        -------------
        a : Array-like
            The array to create the view on

        window : tuple or int
            If int, the size of the window in `axis`, or in all dimensions if 
            `axis == None`

            If tuple, the shape of the desired window.  `window.size` must be:
                equal to `len(axis)` if `axis != None`, else 
                equal to `len(a.shape)`, or 
                1

        step : tuple, int or None
            The offset between consecutive windows in desired dimension
            If None, offset is one in all dimensions
            If int, the offset for all windows over `axis`
            If tuple, the step along each `axis`.  
                `len(step)` must me equal to `len(axis)`

        axis : tuple, int or None
            The axes over which to apply the window
            If None, apply over all dimensions
            if tuple or int, the dimensions over which to apply the window

        generator : boolean
            Creates a generator over the windows 
            If False, it will be an array with 
                `a.nidim + 1 <= a_view.ndim <= a.ndim *2`.  
            If True, generates one window per .next() call
        
        readonly: return array as readonly

        Returns
        -------

        a_view : ndarray
            A windowed view on the input array `a`, or a generator over the windows   

        """
        ashp = np.array(a.shape)
        if axis != None:
            axs = np.array(axis, ndmin = 1)
            assert np.all(np.in1d(axs, np.arange(ashp.size))), "Axes out of range"
        else:
            axs = np.arange(ashp.size)

        window = np.array(window, ndmin = 1)
        assert (window.size == axs.size) | (window.size == 1), "Window dims and axes don't match"
        wshp = ashp.copy()
        wshp[axs] = window
        assert np.all(wshp <= ashp), "Window is bigger than input array in axes"

        stp = np.ones_like(ashp)
        if step:
            step = np.array(step, ndmin = 1)
            assert np.all(step > 0), "Only positive step allowed"
            assert (step.size == axs.size) | (step.size == 1), "step and axes don't match"
            stp[axs] = step

        astr = np.array(a.strides)

        shape = tuple((ashp - wshp) // stp + 1) + tuple(wshp)
        strides = tuple(astr * stp) + tuple(astr)

        as_strided = np.lib.stride_tricks.as_strided
        a_view = np.squeeze(as_strided(a, 
                                     shape = shape, 
                                     strides = strides, writeable=not readonly))
        
        return a_view


def extract_windows(signal, sfreq, wsize, step=30, pad=True):
    """ 
    Extract windows from a signal of a given window size with striding step
    
    :param sfreq:  the sampling frequency of the signal
    :param wsize:  the size of the window
    :param step:   stepize of the window extraction. If None, stride=wsize
    :param pad:    whether to pad the array such that there are exactly 
                   len(signal)//stride windows (e.g. same as hypnogram)
    """ 
    assert signal.ndim==1
    if step is None: step = wsize
    step *= sfreq
    wsize *= sfreq
    assert len(signal)>=wsize, 'signal is shorter than window size'
    n_step = len(signal)//step

    if pad:
        padding = (wsize//2-step//2)
        signal = np.pad(signal, [padding, padding], mode='reflect')
    windows = _window_view(signal, window=wsize, step=step, readonly=True)

    if pad:
        assert n_step == len(windows), f'unequal sizes {n_step}!={len(windows)}'
    return windows

def extract_RR_windows(T_RR, RR, wsize, step=30, pad=True,
                       expected_nwin=None):
    """ 
    Extract windows from a list of RR intervals of a given window size 
    with striding step. The windows are centered around the step borders.
    E.g. step=30, the first window will be centered around second 15,
    iff padding is activated.
    
    :param T_RR: the peak locations
    :param RR: a list of differences between RR peaks, e.g. [1.4, 1.5, 1.4]
    :param wsize:  the size of the window
    :param step: stepize of the window extraction. If None, stride=wsize
    :param pad:    whether to pad the array such that there are exactly 
                   len(signal)//stride windows (e.g. same as hypnogram)
    """ 
    
    # last detected peak should give roughly the recording length.
    # however, this is not always true, ie with a flat line at the end
    if T_RR[0]>1000: 
        raise ValueError(f'First peak at second {T_RR[0]}, seems wrong. Did you substract seconds after midnight?')
    record_len = int(T_RR[-1])
    if expected_nwin is None:
        expected_nwin = record_len//30 
    
    # this array gives us the position of the RR at second x
    # e.g. seconds_idxs[5] will return the RR indices starting at second 5.
    second_idxs = []
    c = 0 
    for i in range(record_len):
        while i>=T_RR[c]:
            c+=1
        second_idxs.append(c)
    second_idxs = np.array(second_idxs)
    assert record_len==len(second_idxs), f'record len={record_len}, but seconds array is {len(second_idxs)}'

    if step==None: step = wsize

    # pad left and right by reflecting the array to have
    # the same number of windows as e.g. hypnogram annotations
    if pad: 
        # this is how many seconds we need to add at the beginning and end
        pad_len_sec = (wsize//2-step//2)
        if pad_len_sec>0:
            # these are the positions of this second in the RR array
            pad_rr_i_l = second_idxs[pad_len_sec]
            pad_rr_i_r = second_idxs[len(second_idxs)-pad_len_sec]
            
            # These are the values that we want to add before and after
            pad_rr_l = RR[:pad_rr_i_l]
            pad_rr_r = RR[pad_rr_i_r:]
            RR = np.hstack([pad_rr_l, RR, pad_rr_r])
            
            # we also need to re-adapt the second_idxs, to know at which second
            # which RR is now.
            pad_sec_l = second_idxs[:pad_len_sec]
            pad_sec_r = second_idxs[-pad_len_sec:] 
            pad_sec_r = pad_sec_r + pad_sec_r[-1] -  pad_sec_r[0] + pad_sec_l[-1]+2
            second_idxs = second_idxs + pad_sec_l[-1] + 1
            second_idxs = np.hstack([pad_sec_l, second_idxs, pad_sec_r])
        
    # assert second_idxs[-1]==len(RR)-1
    # these are the centers of the windows, exactly between two step boundaries
    windows = []
    # n_windows = int(len(second_idxs)//step)-wsize//step+1
    for i in range(expected_nwin):
        # get RR values for this window
        if i*step>=len(second_idxs) or i*step+wsize>=len(second_idxs):
            windows.append(np.array([]))
            continue
        idx_start = second_idxs[i*step]
        idx_end = second_idxs[i*step+wsize]
        wRR = RR[idx_start:idx_end]
        windows.append(wRR)
    # assert expected_nwin==len(windows)
    return windows
   
   
def artefact_detection(T_RR, RR, wsize=30, step=30, expected_nwin=None):
    """
    Scans RR interval arrays for artefacts.
    
    Returns an array with wsize windows and the number of 
    artefacts in this window. 
    
    good examples: 
        flat line with no flat line: 106_06263, 107_13396
        flat line once in a while: 659_88640
        todo: look for tiny ecg as well
        
    The following artefact correction procedure is applied on 30s epochs:
        2. If any RR is > 2 (==HR 30, implausible), discard
        3. If n_RRi == 0: Epoch ok.
        4. If n_RRi<=2: Epoch ok. (keep RRi)
        5. If n_RRi==3: 
        If n_RRi are not consecutive: Epoch ok (keep RRi)
        else: discard
        6. If n_RRi>=4: Discard epoch.

    The following artefact correction procedure is applied on 300s epochs:
        If more than 5% is corrected: discard
        else: keep
        
    """
    if step is None: step = wsize
    assert wsize in [30, 300], 'Currently only 30 and 300 are allowed as artefact window sizes, we didnt define other cases yet.'

    idxs = extract_RR_windows(T_RR, np.arange(len(RR)), wsize, step=step, expected_nwin=expected_nwin)
 
    # RR_pre is before correction
    # RR_post is after correction (as coming directly from Kubios)
    RR_orig = np.diff(T_RR)
    RR_corr = RR
    windows_RR_orig =  [RR_orig[idx] if len(idx)>0 else [] for idx in idxs]
    windows_RR_corr = [RR_corr[idx] if len(idx)>0 else [] for idx in idxs]
    assert len(windows_RR_orig)==len(windows_RR_corr)
    
    art = []
    for w_RR_orig, w_RRi_corr in zip(windows_RR_orig, windows_RR_corr):

        assert len(w_RR_orig) == len(w_RRi_corr)
        # 2. HR<30, seems odd
        # but also any corrected RR larger than 2 or smaller than 0.4 seems wrong
        if len(w_RR_orig)<wsize/2 or (w_RRi_corr>2).any() or (w_RRi_corr<0.4).any():
            art.append(True)
            continue
        else:
            diff = np.where(~np.isclose(w_RR_orig, w_RRi_corr))[0]
            percentage = len(diff)/len(w_RR_orig)
            # 5. special case, are they consecutive or not?
            if wsize==30 and len(diff)==3:
                # are the consecutive? Then the summary of their distance should be 3
                if np.sum(diff)==3:
                    art.append(True)
                    continue
            # else more than 3 corrected values and wsize=30, discard
            elif wsize==30 and len(diff)>3:
                art.append(True)
                continue
            # if wsize is 300, we only look if percentage of corrected is > 0.05
            elif wsize==300 and percentage>0.05:
                art.append(True)
                continue

        # if we reach this far, no artefact has been detected
        art.append(False)
        
    art = np.array(art)
    assert len(art)==len(windows_RR_corr)
    return art

    
#%% main
if __name__=='__main__':
    import sleep
    p = sleep.Patient('Z:/NT1-HRV-unisens/000_16462')
    RR_windows = p.get_feat('identity', only_clean=False, cache=False)
    RR_windows = [x for x in RR_windows]
