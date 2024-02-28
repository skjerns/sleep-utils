# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 09:47:39 2018
@author: SimonKern
"""
import mne
import warnings
import scipy
import numpy as np
from PIL import Image
from scipy.signal import correlate, get_window, iirnotch
from scipy.signal import butter, lfilter, convolve2d, welch, hilbert
from scipy.stats import zscore


# window types used as tapers without parameters
WTYPES = ['boxcar', 'triang', 'blackman', 'hamming', 'hann', 'bartlett', 
         'flattop', 'parzen', 'bohman', 'blackmanharris', 'nuttall',
         'barthann']

###### Begin of functions


def dig2phys(signal, dmin, dmax, pmin, pmax):
    """converts digital to analogue signals"""
    m = (pmax-pmin) / (dmax-dmin)
    physical = m * signal
    return physical

def phys2dig(signal, dmin, dmax, pmin, pmax):
   """converts analogue to digital signals"""
   m = (dmax-dmin)/(pmax-pmin) 
   digital = (m * signal).astype(np.int, copy=False)
   return digital


def rfft_mag(signals):
    """
    Returns the magnitude of the fft of a signal: sqrt(real^2+imag^2)
    Will return only the real part of frequencies (arraylength//2), see rfft
   
    :param signals: 1D or 2D array
    :returns: the magnitude of the frequencies in the signal
    """

    w = rfft(signals) 
    mag = np.abs(w) # same as np.abs(w)
    return mag


def rfft(signals, *args, **kwargs):
    """
    wrapper for rFFT, mainly for checking differences of Python and C
    Input should be of length power of 2 for optimization purposes.
    Returns the arraylength // 2 (unlike rfft, which does weird wlen//2 +1)
    
    :param signals: a 1D or 2D signal, where each row contains one window of data
    :returns: complex array of fft results, 
              Note: in JSON I save a complex array as two lists with real and imaginary part
    """
    if not isinstance(signals, np.ndarray): 
        signals = np.asarray(signals)    
    wlen = signals.shape[-1]
    if not ((wlen & (wlen - 1)) == 0): 
        warnings.warn('Input array should have length power of 2 for fft, but is: {}'.format(wlen))
    w = scipy.fftpack.fft(signals)[...,:signals.shape[-1]//2]
    assert w.shape[-1] == signals.shape[-1]//2, 'w.shape = {}'.format(w.shape)
    return w




def abs2(x):
    """highly optimized version for magnitude taking.
       if there are problems, just remove the decorator line @numba.vectorize(...)
       if working with float32, need to change signature to inlcude ,numba.float32(numba.complex64)
    """
    return x.real**2 + x.imag**2


def welchs(signals, nperseg=None, overlap=0.0, taper_name='boxcar'):
    """
    A minimal implementation of Welch's method, optimized ~10x faster than previously, 
    even faster than scipy.signal.welch().
    This is also equivalent to the normal rfft(), if used with standard parameters
    
    If nperseg=0 we have a standard fft.
    If nperseg>0 and overlap=0 we have Barlett's methd
    If nperseg>0 and overlap>0 we have Welch's method (use a taper in this case)
        
    :param signals: The signals on which to perform Welchs (1D or 2D)
    :param nperseg: How many samples per segment for the calculation of the FFT
    :param overlap: How many samples should overlap between segments (int or float)
                    if float: is interpreted as ratio of nperseg
                    eg overlap 0.5 = 0.5*nperseg
    :param taper_name: Which taper to use, if overlap>0, it is recommended to use a taper
    :returns: The Welch's frequency estimate of the signal as magnitudes
    """
    from sklearn import feature_extraction

    signals = np.atleast_2d(signals)
    siglen = signals.shape[-1]
    
    if nperseg is None: nperseg = siglen
    if isinstance(overlap, float): overlap = overlap*nperseg
    if siglen%nperseg!=0: warnings.warn('window length should be a multiple of npseg')

    # get window only once, else it will be created for each segment, swallowing computation
    taper = get_window(taper_name, nperseg, False).astype(int) if taper_name not in ['boxcar', None] else None
    
    # this step we have to take to get to the next segment
    step = int(nperseg - overlap)
    n_segments    = (siglen-nperseg)//step +1
    
    # fastest way to extract patches using np.lib.stride_tricks.as_strided via sklearn
    fft_segments = feature_extraction.image.extract_patches(signals, patch_shape=(len(signals),nperseg), extraction_step=step)
    fft_segments = fft_segments.reshape([n_segments, len(signals), nperseg])
    fft_segments = np.swapaxes(fft_segments, 0,1)

    # apply taper
    if taper is not None:
        fft_segments  = np.multiply(fft_segments, taper)
        
    # apply fft and take squared magnitude
    w = rfft(fft_segments)
    w_mag = abs2(w)
    fft_result = np.mean(w_mag, axis=1)
    return fft_result


def welchs_unoptimized(signals, nperseg=None, overlap=0.0, taper_name='boxcar'):
    """
    A minimal implementation of Welch's method, unoptimized, should be equivalent to welchs()
    It's kept here for archive purposes and for C-compatibility, also to check 
    the optimized version in case of changes.
    
    If nperseg=0 we have a standard fft.
    If nperseg>0 and overlap=0 we have Barlett's methd
    If nperseg>0 and overlap>0 we have Welch's method (use a taper in this case)
    
    :param signals: The signals on which to perform Welchs (1D or 2D)
    :param nperseg: How many samples per segment for the calculation of the FFT
    :param overlap: How many samples should overlap between segments (int or float)
                    if float: is interpreted as ratio of nperseg
                    eg overlap 0.5 = 0.5*nperseg
    :param taper_name: Which taper to use, if overlap>0, it is recommended to use a taper
    :returns: The Welch's frequency estimate of the signal as magnitudes
    """
    signals = np.atleast_2d(signals)
    
    siglen = signals.shape[-1]
    if nperseg is None: nperseg = siglen
    if isinstance(overlap, float): overlap = overlap*nperseg
    if siglen%nperseg!=0: warnings.warn('window length should be a multiple of npseg')

    step = int(nperseg - overlap) # distance between windows that we extract
    n_segments = (siglen-nperseg)//step + 1 # this is how many windows we can extract from the signal
    fft_signals = []
    # loop through the signals (in C you can start at the second loop as we only have 1D arrays)
    for sig in signals:
        # collect one fft result per segment
        # in fft_mean we save a running mean of the fft segments
        fft_mean = np.zeros(nperseg//2) 
        # loop through the signal in step sizes that allow for overlap
        for i in range(0, siglen, step):
            # only take full segments
            if i+nperseg>siglen: 
                break 
            # extract segment
            segment = sig[i:i+nperseg]
            # taper segment if necessary
            if taper_name is not None:
                segment = taper(segment, taper_name)
            # take FFT and add to fft results for segments
            w = rfft(segment)
            # this is the same as saving all segments and taking a mean later,
            # but prevents us from having to keep intermediate results
            fft_mean += (np.abs(w)**2) / n_segments 
        # take the mean spectral values for all segments
        fft_signals.append(fft_mean)
    return np.array(fft_signals)


def moving_average(signals, N):
    signals = np.atleast_2d(signals).astype(dtype=int, copy=False)
    return convolve2d(signals, np.ones((1, N))/N, mode='valid')
    

def binnedmean(signals, target_size=None, bin_edges=None, func=None):
    """
    cuts a signal in n parts, applies a function to each part (e.g. mean)
    this way we can go from an array of length M to an array of length N<M
    by taking the mean of the elemts of each part, where the number of parts is N
        
    :param target_size: The number of bins that we should interpolate to
    :bin_edges: custom bin edges, if targed size is not present. Bin edges is a list of tuples
                each tuple represents the edges of the bin [inc, excl].
    :func: a function that should be applied, standard is mean. 
           this function needs to operate only on the first dimension
    """
    # we only deal with 2D data in the given DTYPE
    signals = np.atleast_2d(signals)

    assert signals.ndim<3
    # signal length
    slen = signals.shape[1]
    # if there are no bin edges provided, we are creating the bin edges
    # using the target_size.
    # that means, we create target_size bins, that are equally spaced
    # around the signal
    if bin_edges is None and target_size is not None:
        bin_edges = np.linspace(0, slen, target_size+1, dtype=np.int)
        bin_edges = np.repeat(bin_edges,2)[1:-1]
        bin_edges = bin_edges.reshape([-1,2])
    # if no function is provided, we take the mean of each bin
    if func is None:
        func = lambda s: np.mean(s, axis=1)
    
    new_array = np.zeros([signals.shape[0], len(bin_edges)])
    
    # now we run over the original array and always take all values of each bin
    # in the original signal. then we apply the function
    # that is provided on that bin and write the result to a new array
    for i, [e1,e2] in enumerate(bin_edges):
        bin_entries = signals[:,e1:e2]
        reduced = func(bin_entries)
        new_array[:,i] = reduced
        
    return new_array
   

def bands2bins(bands, window_len, sfreq):    
    """
    A function that turns frequency bands into bin-edges to be used to extract these bands
    from a fft function
    
    :param bands: a list of 2-tuples, each 2-tuple containing a frequency lower and upper bound
    :param window_len: the length of the window that the FFT will be applied to
    :param sfreq: the sampling frequency of the signal
    """
    sfreq = int(sfreq)
    window_len=int(window_len)
    round2=lambda x,y=None: round(x+1e-15,y)
    bins = []
    for l,h in bands:
        bin_lower = int(round2(l*window_len/sfreq)) # np.floor rounds down
        bin_upper = int(round2(h*window_len/sfreq))  # np.ceil rounds up
        bins.append([bin_lower, bin_upper])
    return bins


def create_specband_bin_edges(window_len=None, sfreq=None):
    """
    Given a FFT-window length and a sampling frequency, 
    compute the bin edges for common brain frequencies
    :param window_len: length in sample numbers, if None, will just return bands and not bins
    :param sfreq: sampling frequency, if None, will just return bands and not bins
    :returns: edges for [slowos, delta, theta, alpha1, spindle, beta1, beta2, gamma]
    
    BRAIN_BAND_NAMES_ = ['so', 'delta', 'theta', 'alpha1', 'alpha2', 'beta1', 'beta2', 'gamma']
    
    """
    if sfreq is None: sfreq=window_len
    
    slowos   = [0, 1]
    delta   = [1, 4]
    theta   = [4, 8]
    alpha1  = [8, 10]
    spindle = [10, 14]
    beta1   = [13, 16]
    beta2   = [16, 20]
    gamma   = [20, 35]
    
    # this is the order of the bands
    # result is a list of tuples, where each tuple is the bin edge
    bands = [slowos, delta, theta, alpha1, spindle, beta1, beta2, gamma]
    if window_len is None and sfreq is None: return bands
    # calculate the bin positions corresponding to frequency edges
    bins = bands2bins(bands, window_len, sfreq)
    return bins # this is a list of tuples, e.g. [(0,30),(30,60),(60,120)]

     

def taper(signals, window_name):
    """
    filters/tapers given signals or signal with a window with window-name. allowed names are:
        ['boxcar', 'triang', 'blackman', 'hamming', 'hann', 'bartlett',
        'flattop', 'parzen', 'bohman', 'blackmanharris', 'nuttall',
        'barthann'] 
                
    :param signals: A 1D signals or a 2D array with signals x samples
    :param window_name: the name of the taper
    :returns: the tapered signals
    """  
    if window_name in [None, '', 'boxcar']: return signals
    if window_name.lower() not in ['hamming', 'hann', 'blackman', 'blackmanharris']:
        warnings.warn('Up to this point, this window is not tested in C code: {}.'.format(window_name))
    if isinstance(signals, list): signals = np.array(signals)
    assert signals.ndim in [1,2], 'signals must be 1D or 2D'
    # how long is the signal?
    wlen = signals.shape[0] if signals.ndim==1 else signals.shape[1]
    # get the window function using scipy
    window = get_window(window_name, wlen, False)
    # filter our input signal
    tapered = signals*window  
    return tapered


def extract_windows(signal, length, distance=1.0, group_n=0, group_d=0, hypno=None, taper_name=None):
    """
    From a signal, extract windows every N seconds for a length of L.
    Can be grouped in groups on group_n, with distance between last samplepoint of one group
    and the first sample of the next group being group_d.
    only complete groups will be returned    
    
    :param signal: the signal to take windows from
                   if the signal is 2D, the format signaln x samples is assumed
    :param length: The length of each window in sample space
    :param distance: the distance between each windows
                     if distance is an int, it will be interpreted as a spacing in sample space
                     if distance is a float, it will be interpreted as relative to the length
                     e.g. distance=0.5 will produce windows with overlap of 50%
    :param group_n: how many windows to extract before leaving a space to the next group of windows
    :param group_d: how much space between batches
                     if distance is an int, it will be interpreted as a spacing in sample space
                     if distance is a float, it will be interpreted as relative to the length
                     e.g. group_d=0.5 will produce windows in groups with distance between groups 
                     of 50% window length 
    :param hypno: a hypnogram annotation. It will automatically be inferred what spacing the 
                  hypnogram uses and what annotation frequency
    """
    # pass through if length is None
    if length is None: 
        return (signal, hypno)
    
    assert (group_n==0 and group_d==0) or (group_n>0 and group_d>0), \
            'group parameters must both be supplied'
    assert distance>0, 'distance must be positive'
    assert length>0, 'length must be positive'
    if hypno is not None: assert type(hypno) is np.ndarray, 'must be numpy array'
    assert type(signal) is np.ndarray, 'must be numpy array'
    
    siglen = len(signal) if signal.ndim==1 else signal.shape[1]
    distance = distance if type(distance) is int else int(distance*length)
    group_d  = group_d if type(group_d) is int else int(group_d*length)

    assert length<=siglen, 'length must be smaller than siglen'
   
    if signal.ndim == 1: signal = [signal] # to make the iterator work in 1D case
    
    
    # inefficient implenentation, but makes sure everything is correct
    
    all_sigs = []
    for subsig in signal:
        i = 0
        g = 0 # group counter
        sigs = []
        while i <= siglen-length:
            s = subsig[i:i+length]
            sigs.append(s)
            i+=distance
            g+=1
            if g>=group_n and group_n>0:
                i+=group_d+length-distance
                g=0
        all_sigs.append(np.array(sigs))
        
    all_sigs = np.array(all_sigs)
    if all_sigs.shape[0]==1: all_sigs = all_sigs.reshape(-1, length)
    
    # match a new hypnogram to the extracted epochs
    if hypno is not None:
        s_per_hypno = siglen/len(hypno)
        i = 0
        g = 0 # group counter
        new_hypno = []
        while i <= siglen-length:
            h = hypno[int(i/s_per_hypno)]
            new_hypno.append(h)
            i+=distance
            g+=1
            if g>=group_n and group_n>0:
                i+=group_d+length-distance
                g=0
    windows   = np.array(all_sigs, copy=False)
    if taper_name is not None: windows = taper(windows, taper_name)
    if hypno is not None: new_hypno = np.array(new_hypno, dtype = hypno.dtype) 
    return windows if hypno is None else (windows, new_hypno)


def correlate_specto(signal1, signal2, sfreq):
    """
    create a cross-correlation of two signals using the spectogram
    """
    
    sfreq1 = 256
    sfreq2 = 200
    n_samples1 = int(sfreq1)
    n_samples2 = int(sfreq2)
    
    # get fft results 
    signal1_trunc = signal1[:int((len(signal1)//n_samples1)*(n_samples1))].reshape([-1,n_samples1])
    signal2_trunc = signal2[:int((len(signal2)//n_samples2)*(n_samples2))].reshape([-1,n_samples2])
    
    # get FFT via Welchs
    freq1, spec1 = welch(signal1_trunc, fs=sfreq1, window='boxcar', nperseg=sfreq1, noverlap=0)
    freq2, spec2 = welch(signal2_trunc, fs=sfreq2, window='boxcar', nperseg=sfreq2, noverlap=0)
    
    # truncate from 2 to 15 Hz
    spec1 = spec1[:,np.argmax(freq1>2):np.argmax(freq1>15)]
    spec2 = spec2[:,np.argmax(freq1>2):np.argmax(freq1>15)]
  
    # transform to dB
    spec1 = (np.log10(spec1+1)*20).T
    spec2 = (np.log10(spec2+1)*20).T

    # make row-wise correlation
    corr = np.array([correlate(zscore(row1), zscore(row2)) for row1, row2 in zip(spec1,spec2)]).mean(0)
    return corr


def coeff_var_env(signal, sfreq, band=(0.5, 4), mid_sec=30):
    """Coefficient of the variance of the envelope (CVE) calculation
    
    as defined in  https://doi.org/10.1016/j.neuroimage.2018.01.063
    
    :param epoch: one epoch of data, 1D data as array
    :type epoch: np.ndarray
    :return: CVE of this epoch
    :rtype: TYPE
    """
    sig = np.atleast_2d(signal)
    sig_filtered = np.atleast_2d(bandfilter(sig, sfreq, *band, method='iir',
                                            iir_params={'order':4, 
                                                        'ftype':'butter'}))
    ht_sig_filtered = np.abs(hilbert(sig_filtered))
    
    eeg_envelope = np.sqrt(sig_filtered**2 + ht_sig_filtered**2)
    
    t_excess = int(signal.shape[-1] -  mid_sec*sfreq)//2
    middle_env = eeg_envelope[:, t_excess:-t_excess]
    assert abs(middle_env.shape[1]-mid_sec*sfreq)<2
    
    mean = np.mean(middle_env)
    sd = np.std(middle_env)
    
    cve = sd/(mean*0.523)  # [..] with 0.523 being the value for Gaussian waves
     
    return cve

def get_shift_specto(signal1, signal2, sfreq = 256, w_seconds=1, limit_to=1):
    """
    a more robust version to align two eeg recordings using fourier transformations
    accuracy of this method depends on w_seconds
    
    
    :param signal1: a 1D signal
    :param signal2: a 1D signal
    :param sfreq: the sampling frequency of the signals
    :param w_seconds: how many chunks should be used to create the FFT. This determines the accuracy
    """
    assert signal1.ndim==1
    assert signal2.ndim==1
    assert np.all([len(signal1)>sfreq*60*5]), 'need at least 5 minutes of data'
    if len(signal1)>len(signal2)*2: 
        warnings.warn('signal1 is significantly larger than signal2: \
                      is the sampling frequency the same?')
    if len(signal2)>len(signal1)*2: 
        warnings.warn('signal2 is significantly larger than signal1: \
                      is the sampling frequency the same?')
    
    n_samples = int(w_seconds*sfreq)
    
    # get fft results 
    signal1_trunc = signal1[:int((len(signal1)//n_samples)*(n_samples))].reshape([-1,n_samples])
    signal2_trunc = signal2[:int((len(signal2)//n_samples)*(n_samples))].reshape([-1,n_samples])
    
    # get FFT via Welchs
    freq1, spec1 = welch(signal1_trunc, fs=sfreq, window='blackmanharris', nperseg=sfreq, noverlap=0.667*sfreq)
    freq2, spec2 = welch(signal2_trunc, fs=sfreq, window='blackmanharris', nperseg=sfreq, noverlap=0.667*sfreq)
    
    # truncate from 2 to 15 Hz
    spec1 = spec1[:,np.argmax(freq1>2):np.argmax(freq1>15)]
    spec2 = spec2[:,np.argmax(freq1>2):np.argmax(freq1>15)]
  
    # transform to dB
    spec1 = (np.log10(spec1+1)*20).T
    spec2 = (np.log10(spec2+1)*20).T

    # make row-wise correlation
    corr = np.array([correlate(zscore(row1), zscore(row2)) for row1, row2 in zip(spec1,spec2)]).mean(0)
    half = len(corr)-spec1.shape[1]
    corr[:int(half-sfreq*60**2*limit_to//n_samples)] = 0
    corr[int(half+sfreq*60**2*limit_to//n_samples):] = 0
    peak = corr.argmax()
    shift = (len(corr)-spec1.shape[1]- peak) * n_samples
    return -shift


def get_shift_spindle(signal1, signal2, sfreq, limit_to=1):
    """
    Taking two frontal channels, will try to recover the shift
    via aligning the spindle band. This should filter out a lot
    of other non-related noise and be quite robust for sleep recordings
    
    Shift will be searched for within 1h of the recording start
    
    :param signal1: a 1D signal
    :param signal2: a 1D signal
    :param limit_to: limit to +- this hour
    """
    assert signal1.ndim==1
    assert signal2.ndim==1
    if len(signal1)>len(signal2)*2: 
        warnings.warn('signal1 is significantly larger than signal2: \
                      is the sampling frequency the same?')
    if len(signal2)>len(signal1)*2: 
        warnings.warn('signal2 is significantly larger than signal1: \
                      is the sampling frequency the same?')
        
    sigma1 = butter_bandpass_filter(signal1/signal1.max(), 7, 14, sfreq, 5)
    sigma2 = butter_bandpass_filter(signal2/signal2.max(), 7, 14, sfreq, 5)
    
    corr = correlate(sigma1, sigma2)
    # sometimes there are some weird correlations if the signals are shifted
    # extremely. We assume that the recording alignment must lie within 
    # 1 h of either recording start
    half = len(corr)//2
    corr[:half-sfreq*60**2*limit_to] = 0
    corr[half+sfreq*60**2*limit_to:] = 0
    peak = corr.argmax()
    shift = len(signal2) - peak -1 
    print(shift/sfreq, ' seconds')
#    plt.plot(corr)
    return -shift    


    
def get_shift(signal1, signal2, sfreq, limit_to=1):
    """
    find the shift of signal1 to signal2, used for aligning two signals
    
    Data must have the same sampling frequency
    
    The output will be how much we need to roll signal2 to match signal1
    np.roll(signal2, shift)
    
    :param signal1: a 1D signal
    :param signal2: a 1D signal
    """
    assert signal1.ndim==1
    assert signal2.ndim==1
    if len(signal1)>len(signal2)*2: 
        warnings.warn('signal1 is significantly larger than signal2: \
                      is the sampling frequency the same?')
    if len(signal2)>len(signal1)*2: 
        warnings.warn('signal2 is significantly larger than signal1: \
                      is the sampling frequency the same?')
    if signal1.max()>10: signal1 = zscore(signal1, axis=None)
    if signal2.max()>10: signal2 = zscore(signal2, axis=None)
    
    signal1 = butter_bandpass_filter(signal1/signal1.max(), 1, 30, sfreq, 5)
    signal2 = butter_bandpass_filter(signal2/signal2.max(), 1, 30, sfreq, 5)
    
    corr = correlate(signal1, signal2)
    half = len(corr)//2
    corr[:half-sfreq*60**2*limit_to] = 0
    corr[half+sfreq*60**2*limit_to:] = 0
    peak = corr.argmax()
    shift = len(signal2) - peak -1 
    return -shift

def get_correlation(signal1, signal2, sfreq, llim=2, ulim=20):
    """
    find the shift of signal1 to signal2, used for aligning two signals
    
    Data must have the same sampling frequency
    
    The output will be how much we need to roll signal2 to match signal1
    np.roll(signal2, shift)
    
    :param signal1: a 1D signal
    :param signal2: a 1D signal
    """
    assert signal1.ndim==1
    assert signal2.ndim==1
    if len(signal1)>len(signal2)*2: 
        warnings.warn('signal1 is significantly larger than signal2: \
                      is the sampling frequency the same?')
    if len(signal2)>len(signal1)*2: 
        warnings.warn('signal2 is significantly larger than signal1: \
                      is the sampling frequency the same?')
    if signal1.max()>10: signal1 = zscore(signal1, axis=None)
    if signal2.max()>10: signal2 = zscore(signal2, axis=None)
    
    signal1 = butter_bandpass_filter(signal1, 0.1, 30, sfreq, 5)
    signal2 = butter_bandpass_filter(signal2, 0.1, 30, sfreq, 5)
    
    corr = correlate(np.abs(signal1), np.abs(signal2))
    return corr


def resample(data, o_sfreq, t_sfreq):
    """
    resample a signal using MNE resample functions
    This automatically is optimized for EEG applying filters etc
    
    :param raw:     a 1D data array
    :param o_sfreq: the original sampling frequency
    :param t_sfreq: the target sampling frequency
    :returns: the resampled signal
    """
    if o_sfreq==t_sfreq: return data
    raw = np.atleast_2d(data)
    n_jobs = min(len(raw), 8)
    ch_names=['ch{}'.format(i) for i in range(len(raw))]
    info = mne.create_info(ch_names=ch_names, sfreq=o_sfreq, ch_types=['eeg'])
    raw_mne = mne.io.RawArray(raw, info, verbose='WARNING')
    resampled = raw_mne.resample(t_sfreq, n_jobs=n_jobs, verbose='WARNING')
    new_raw = resampled.get_data().squeeze()
    return new_raw.astype(raw.dtype, copy=False)

def resize(array, target_size):
    """
    Resize a 1D array containing a signal/
    or a 2D array of several signasl as rows of any type (int/float) 
    by taking nearest neighbours to fill gaps within the rows
    
    :param array: any type of array
    :param target_size: the target size of the last dimension
    """
    array = np.atleast_2d(array)
    new_array = []
    for row in array:
        pil_row = Image.fromarray(row.reshape([1,-1]))
        pil_resized = pil_row.resize((target_size,1), Image.NEAREST)
        new_array.append(np.array(pil_resized).squeeze())
    return np.array(new_array).squeeze()
    
    

def bandfilter(data, sfreq, l_freq, h_freq, **kwargs):
    """Use mne.io.RawArray.filter to create a bandpath filter for this signal
    
    Can be either 1D or 2D signal
    
    :param raw: the signal
    :param sfreq: the sampling frequency of the signal
    :param l_freq: the lower frequency
    :param h_freq: the higher frequency
    """
    assert data.ndim<3, 'Data must be 2D or 1D'
    ndim = data.ndim
    data = np.atleast_2d(data)
    ch_names = [f'ch{i}' for i,_ in enumerate(data)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg'])
    raw = mne.io.RawArray(data, info, verbose='WARNING', copy='both')
    raw = raw.filter(l_freq, h_freq, verbose='WARNING', **kwargs)
    new_data = raw.get_data()
    return new_data.squeeze() if ndim==1 else new_data

def notch(data, sfreq=256, f0=50):
    fs = sfreq  # Sample frequency (Hz)
    Q = 30.0  # Quality factor
    w0 = f0/(fs/2)
    b, a = iirnotch(w0, Q)
    filtered = lfilter(b,a, data)
    return filtered.astype(data, copy=False)

########### unchecked functions!! be aware

def highpass(signal, sfreq, hz):
    fc = hz/sfreq  # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
    b = 0.08  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
    N = int(np.ceil((4 / b)))
    if not N % 2: N += 1  # Make sure that N is odd.
    n = np.arange(N)
     
    # Compute a low-pass filter.
    h = np.sinc(2 * fc * (n - (N - 1) / 2.))
    w = np.blackman(N)
    h = h * w
    h = h / np.sum(h)
     
    # Create a high-pass filter from the low-pass filter through spectral inversion.
    h = -h
    h[(N - 1) // 2] += 1
    return np.convolve(signal, h)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def lowpass(signal, sfreq, hz): 
    fc = hz/sfreq  # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
    b = 0.08  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
    N = int(np.ceil((4 / b)))
    if not N % 2: N += 1  # Make sure that N is odd.
    n = np.arange(N)
     
    # Compute sinc filter.
    h = np.sinc(2 * fc * (n - (N - 1) / 2.))
     
    # Compute Blackman window.
    w = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + \
        0.08 * np.cos(4 * np.pi * n / (N - 1))
     
    # Multiply sinc filter with window.
    h = h * w
     
    # Normalize to get unity gain.
    h = h // np.sum(h)
    return np.convolve(signal, h)