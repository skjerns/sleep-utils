#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 11:20:34 2024

Create an A4 summary for patients including
- hypnogram
- summary of sleep stages
- latencies

@author: simon.kern
"""
import os
import json
import mne
import matplotlib.pyplot as plt
import sleep_utils
import numpy as np
import pandas as pd
from sleep_utils.plotting import plot_hypnogram
from sleep_utils.external import appdirs
from sleep_utils.plotting import choose_file, plot_hypnogram
from sleep_utils.tools import infer_hypno_file, read_hypno, infer_psg_file
# plt.rcParams['text.usetex'] = True

config_dir = appdirs.user_config_dir('sleep-utils')
config_file = os.path.join(config_dir, 'last_used.json')
os.makedirs(config_dir, exist_ok=True)

def split_string_at_spaces(desc, max_length=20):
    words = desc.split()
    chunks = []
    current_chunk = ""

    for word in words:
        if len(current_chunk) + len(word) + 1 <= max_length:
            if current_chunk:
                current_chunk += " " + word
            else:
                current_chunk = word
        else:
            chunks.append(current_chunk)
            current_chunk = word

    if current_chunk:
        chunks.append(current_chunk)

    return '\n'.join(chunks)

def config_load():
    if not os.path.isfile(config_file):
        return {}
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def config_save(config):
    with open(config_file, 'w') as f:
        config = json.dump(config, f)

model = 'U-Sleep v2.0' # which usleep model to use

config = config_load()
prev_dir = config.get('prev_dir', None)

print('Please select hypnogram file in GUI for loading..')
hypno_files = choose_file(default_dir = prev_dir, exts=['txt'], multiple=True,
                         title='Choose (multiple) files to include (select more by pressing CTRL)')

# try to guess the original data file
files = [infer_psg_file(hypno_file) for hypno_file in hypno_files]
participant_id = os.path.basename(files[0]).split('_')[0]

config['prev_dir'] = os.path.dirname(hypno_files[0])
config_save(config)

print('\n### Loading data')
raws = [mne.io.read_raw(file, preload=False) for file in files]
hypnos = [read_hypno(hypno_file) for hypno_file in hypno_files]

report_dir = f'{os.path.dirname(files[0])}/report/'
os.makedirs(report_dir, exist_ok=True)

#%% plot hypnograms & stages pie chart
fig1, ax1 = plt.subplots(1, 1, figsize=[10, 3], dpi=200)
fig2, ax2 = plt.subplots(1, 1, figsize=[5, 5], dpi=200)

night_names = ['Gewöhnungsnacht', 'Nacht 1', 'Nacht 2']

hypno_pngs = []
dist_pngs = []
summaries = []

for raw, hypno in zip(raws, hypnos):
    basename = os.path.basename(os.path.splitext(raw.filenames[0])[0])
    night = night_names[int(basename[6])]  # night id at position 6 of filename

    ax = ax1
    ax.clear()
    starttime = raw.info["meas_date"].replace(microsecond=0, tzinfo=None) # remove timezone and millisecond info
    plot_hypnogram(hypno, ax=ax, starttime=starttime,
                   title=f"Schlafprofil für '{night}', Start um {starttime}")

    # filter annotations that they are not overlapping in the plot
    tdiffs = np.pad(np.diff([annot['onset'] for annot in raw.annotations[1:]]).round(2), [0,1], constant_values=301)
    annotations = [annot for annot, tdiff in zip(raw.annotations[1:], tdiffs) if tdiff>300]

    for i, annot in enumerate(annotations):
        onset = annot['onset']
        duration = annot['duration']
        desc = annot['description'].replace('Comment/', '')
        desc = split_string_at_spaces(desc, 25)
        # dt = annot['orig_time']
        ax.axvspan(onset, onset+duration, alpha=0.2, color='red')
        ax.vlines(onset, *ax.get_ylim(), color='red', linewidth=0.2)
        ax.text(onset, 0.5, desc, color='darkred', horizontalalignment='left' if i==0 else 'right',
                verticalalignment='top', rotation=90, alpha=0.75)

    ax = ax2
    ax.clear()
    starttime = raw.info["meas_date"].replace(microsecond=0, tzinfo=None) # remove timezone and millisecond info
    summary = sleep_utils.hypno_summary(hypno)
    summaries += [pd.Series(summary, name=night)]
    values = [summary['WASO'], summary['min_S1'], summary['min_S2'], summary['min_S3'], summary['min_REM']]
    percent = [summary['perc_W'], summary['perc_S1'], summary['perc_S2'], summary['perc_S3'], summary['perc_REM']]
    labelszip = zip(['\nWach nach Einschlafen', 'S1', 'S2 (leicht)', 'S3 (tief)', 'REM'], values, percent, strict=True)
    if plt.rcParams['text.usetex'] :
        stageslabels = [f'\\textbf{{{l}}}\n({int(m)} min / {p:.01f}%)'for l, m, p in labelszip]
    else:
        stageslabels = [f'{l}\n({int(m)} min / {p:.01f}%)'for l, m, p in labelszip]

    ax.pie(values, labels=stageslabels, explode=[0.025]*5)
    ax.set_title(f"Verteilung '{night}'", fontsize=14, fontweight='bold')

    hypno_png = f'{report_dir}/hypno_{basename}.png'
    dist_png = f'{report_dir}/dist_{basename}.png'
    hypno_pngs += [hypno_png]
    dist_pngs += [dist_png]
    plt.pause(0.1)
    fig1.tight_layout(h_pad=3)
    fig2.tight_layout(h_pad=3)
    plt.pause(0.1)
    fig2.tight_layout(h_pad=3)
    plt.pause(0.1)
    fig1.savefig(hypno_png)
    fig2.savefig(dist_png)


# plt.close(fig1)
# plt.close(fig2)

#%% plot summary table

summaries += [pd.Series({
    'TST': '360-560',  # Total Sleep Time in minutes (6 to 9 hours)
    'TRT': 'N/A',  # Total Recording Time varies
    'TBT': '360-600',  # Total Bed Time in minutes (6 to 10 hours)
    'WASO': '0-30',  # Wake After Sleep Onset in minutes
    'min_S1': '10-30',  # Minutes in stage S1
    'min_S2': '200-300',  # Minutes in stage S2
    'min_S3': '30-90',  # Minutes in stage S3
    'min_REM': '60-120',  # Minutes in REM sleep
    'sum_NREM': '240-420',  # Sum of Minutes in Non-REM
    'perc_W': '1-5%',  # Percentage of wake time
    'perc_S1': '2-8%',  # Percentage of stage S1 time
    'perc_S2': '40-60%',  # Percentage of stage S2 time
    'perc_S3': '10-25%',  # Percentage of stage S3 time
    'perc_REM': '15-30%',  # Percentage of REM sleep time
    'lat_S1': '5-15',  # Latency to stage S1 in minutes
    'lat_S2': '10-20',  # Latency to stage S2 in minutes
    'lat_S3': '20-40',  # Latency to stage S3 in minutes
    'lat_REM': '70-120',  # Latency to REM sleep in minutes
    'sleep_onset_after_rec_start': 'N/A',  # Sleep onset after recording start in minutes
    'sleep_offset_after_rec_start': 'N/A',  # Sleep offset after recording start
    'recording_length': 'N/A',  # Recording length in minutes (7 to 10 hours)
    'awakenings': '0-15',  # Number of awakenings
    'mean_dur_awakenings': '1-5',  # Mean duration of awakenings in minutes
    'FI': 'N/A',  # Fragmentation index
    'sleep_cycles': '3-5',  # Number of sleep cycles
    'stage_shifts': '10-70',  # Number of shifts of sleep stages
    'SQI': '0.8-1.0',  # Sleep quality index (ratio)
    'SE': '85-95%'  # Sleep efficiency percentage
    }, name='Richtwert')]

df_summaries = pd.concat(summaries, axis=1)
full_names = {
    'TST': 'Gesamtschlafzeit',
    'TRT': 'Gesamtaufzeichnungszeit',
    'TBT': 'Gesamte Zeit im Bett',
    'WASO': 'Wachminuten nach erstem Schlafbeginn',
    'min_S1': 'Minuten in Schlafphase S1',
    'min_S2': 'Minuten in Schlafphase S2',
    'min_S3': 'Minuten in Schlafphase S3',
    'min_REM': 'Minuten im REM-Schlaf',
    'sum_NREM': 'Minuten im Non-REM',
    'perc_W': 'Prozent der Wachzeit',
    'perc_S1': 'Prozent der Schlafphase S1',
    'perc_S2': 'Prozent der Schlafphase S2',
    'perc_S3': 'Prozent der Schlafphase S3',
    'perc_REM': 'Prozent der REM-Schlafzeit',
    'lat_S1': 'Latenz bis Schlafphase S1',
    'lat_S2': 'Latenz bis Schlafphase S2',
    'lat_S3': 'Latenz bis Schlafphase S3',
    'lat_REM': 'Latenz bis REM-Schlaf',
    'sleep_onset_after_rec_start': 'Schlafbeginn nach Aufzeichnungsstart',
    'sleep_offset_after_rec_start': 'Schlafende nach Aufzeichnungsstart',
    'recording_length': 'Aufzeichnungsdauer',
    'awakenings': 'Anzahl kurzes Aufwachen',
    'mean_dur_awakenings': 'Durchschnittliche Dauer des Aufwachens',
    'FI': 'Fragmentierungsindex',
    'sleep_cycles': 'Anzahl der Schlafzyklen',
    'stage_shifts': 'Anzahl der Schlafphasenwechsel',
    'SQI': 'Schlafqualitätsindex',
    'SE': 'Schlafeffizienz'
}
df_summaries.index = [full_names[name] for name in df_summaries.index]

#%% create MarkDown file

string = f"# Schlafreport für Proband:in {participant_id}\n"
string += '### Schlafprofil and Schlafphasenverteilung\n\n'
for i, hypno_png in enumerate(hypno_pngs):
    string += f'<img src="./{os.path.basename(hypno_png)}" alt="hypno_{i}" width="100%"/><br><br>'
    # string += f'![hypnogram_{i}](./{os.path.basename(hypno_png)})\n\n'

string += '\n\n<div style="display: flex; justify-content: center; align-items: center;">'

for i, dist_png in enumerate(dist_pngs):
    string += f'<img src="./{os.path.basename(dist_png)}" alt="dist_{i}" width="{1/len(dist_pngs)*100}%"/>'

string += '</div>\n'
string += '<div style="page-break-after: always;"></div><br>'
string += f"\n\n\n### Tabellarische Werte\n\n{df_summaries.to_markdown(stralign='right')}\n"

file_md = f'{report_dir}/report_{participant_id}.md'

with open(file_md, 'w') as f:
    f.write(string)

print(f'Markdown saved to {file_md}')

#%% convert markdown to pdf
# this might only work if you have wkhtmltopdf installed
# install via `conda install wkhtmltopdf `
import markdown2
import pdfkit

file_pdf = os.path.splitext(file_md)[0] + '.pdf'
# Convert Markdown to HTML
html_text = markdown2.markdown(string.replace('./', report_dir),  extras=['tables'])
# Add inline CSS for better table formatting
html_text = f"""
    <html>
    <head>
<meta charset="UTF-8">
<style>
table {{
    width: 70%;
    border-collapse: collapse;
}}
th, td {{
    padding: 6px ; /* Add more space between columns */
    text-align: left;
}}
th {{
    background-color: #bbbbbb; /* Distinct header background color */
    font-weight: bold;
    white-space: nowrap; /* Prevent line breaks in table headers */
}}
tr:nth-child(even) {{
    background-color: #eeeeee; /* Light grey alternate shading */
}}
tr:nth-child(odd) {{
    background-color: #ffffff; /* White for the other rows */
}}

</style>
</head>
<body>
{html_text}
</body>
</html>
"""
# Convert HTML to PDF
assert pdfkit.from_string(html_text, file_pdf), 'converting failed'

print(f'PDF saved to {file_pdf}')
