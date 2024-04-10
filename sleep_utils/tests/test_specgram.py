#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:22:55 2024

@author: simon.kern
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
import sleep_utils

def format_seconds(seconds):
    """Format duration in seconds into HH:MM:SS."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    return '{:02d}:{:02d}:{:02d}'.format(int(hours), int(minutes), int(seconds))

def generate_sine_wave_with_noise(seconds, sfreq, frequency):
    """Generate a sine wave signal."""
    # Calculate the number of samples
    num_samples = int(seconds * sfreq)

    # Create time vector
    t = np.linspace(0, seconds, num_samples, endpoint=False)

    # Generate sine wave
    sine_wave = np.sin(2 * np.pi * frequency * t)
    noise = (1+np.random.rand(len(sine_wave)))
    sine_wave *= noise
    return t, sine_wave

# def test_specgram():
if __name__=='__main__':
    plt.ion()

    seconds = [30, 59, 60*17, 60*69, 60*60*2, 60*60*8]
    np.random.seed(0)
    for i in range(10):
        sfreq = np.random.randint(75, 500)
        sperseg = np.random.randint(3, 30)
        sec = seconds[i]
        t, data = generate_sine_wave_with_noise(sec, sfreq, 10)
        annotations = [{'onset':(1/x)*sec, 'duration':0.2, 'description':f'{x}'} for x in range(2, 5)]
        sleep_utils.specgram_multitaper(data, sfreq=sfreq, sperseg=sperseg, annotations=annotations)
        plt.title(f'{sfreq=}, {sperseg=}, {format_seconds(sec)}')
