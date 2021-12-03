# sleep-utils

A python toolbox for sleep researchers. Plot hypnograms, spectrograms, confusion matrices, PSG summaries

![sample_hypnogram.png](.\assets\d1f7592a94f0f39c4d672c5913e161ec16193458.png)

![spectrogram_multitaper.png](.\assets\c49446ae6d84dee6e13ae14034dd12eb6bbdb48d.png)

#### Install

```
pip install sleep-utils
```

or

```
pip install git+https://github.com/skjerns/sleep-utils
```







#### Functionality

`import sleep_utils`

- hypnograms
  
  - load (`sleep_utils.read_hypno(file)`)
  
  - save(`sleep_utils.write_hypno(hypno, file)`
  
  - plot (`sleep_utils.plot_hypnogram(hypno)`)
  
  - convert (read&save)
  
  - print summary (TST, WASO, ...) (`sleep_utils.hypno_summary(hypno)`)

- spectrograms
  
  - multitaper spectrogram (`sleep_utils.specgram_multitaper(data, sfreq)`)
  
  - welch spectrogram(`sleep_utils.specgram_welch(data, sfreq)`)

- confusion matrix
  
  - plot inter rater confusion matrix (`sleep_utils.plot_confusion_matrix(confmat)`)

- mne-edf
  
  - save MNE to edf (`sleep_utils.write_mne_edf(raw, filename)`)
