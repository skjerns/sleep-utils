# sleep-utils

A python toolbox for sleep researchers. Plot hypnograms, spectrograms, confusion matrices, PSG summaries

<img src="./assets/d1f7592a94f0f39c4d672c5913e161ec16193458.png" title="" alt="sample_hypnogram.png" width="394">

<img src="./assets/c49446ae6d84dee6e13ae14034dd12eb6bbdb48d.png" title="" alt="spectrogram_multitaper.png" width="395">

<img src="md_assets/2022-07-25-13-06-18-image.png" title="" alt="" width="409">

```
Hypnogram summary
{'TRT': 460.5,
 'TST': 444.5,
 'WASO': 16.0,
 'lat_REM': 65.0,
 'lat_S1': 0.0,
 'lat_S2': 2.0,
 'lat_S3': 9.0,
 'min_REM': 109.5,
 'min_S1': 13.5,
 'min_S2': 214.0,
 'min_S3': 107.5,
 'perc_REM': 0.24634420697412823,
 'perc_S1': 0.030371203599550055,
 'perc_S2': 0.4814398200224972,
 'perc_S3': 0.24184476940382452,
 'perc_W': 0.03474484256243214,
 'recording_length': 547.0,
 'sleep_offset_after_rec_start': 534.0,
 'sleep_onset_after_rec_start': 73.5}
```

#### Install

```
pip install sleep-utils
```

or

```
pip install git+https://github.com/skjerns/sleep-utils
```

## Modules

The `sleep-utils` package is organized into several modules, each providing specific functionalities for sleep data analysis.

### `sleep_utils.gui`

This module provides graphical user interface (GUI) components built with `tkinter`. It simplifies tasks like selecting files and folders, and getting user input, making your scripts more interactive.

### `sleep_utils.plotting`

A comprehensive module for creating various plots related to sleep analysis. You can use it to:
- Plot hypnograms
- Generate spectrograms of EEG data
- Visualize confusion matrices for inter-rater reliability
- Plot noise characteristics of your signals

### `sleep_utils.sigproc`

This module contains a collection of functions for signal processing of physiological data. Key features include:
- Filtering and resampling signals
- Detecting artifacts in your data
- Heuristics for spindle detection and other sleep-related events.

### `sleep_utils.tools`

A set of utility functions that make working with sleep data easier. This module provides tools for:
- Reading and writing hypnograms in various formats
- Calculating and summarizing sleep statistics (e.g., TST, WASO, sleep efficiency) based on AASM guidelines.
- Inferring relationships between PSG and hypnogram files.

### `sleep_utils.usleep_utils`

This module provides a convenient wrapper around the [U-Sleep API](https://sleep.ai.ku.dk) for automatic sleep staging. You can use it to:
- Predict sleep stages from EEG/EOG data using the U-Sleep model
- Manage prediction sessions with the API
- It supports both `mne.io.Raw` objects and EDF files as input.