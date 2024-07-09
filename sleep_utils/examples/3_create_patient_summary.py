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
import subprocess
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
fig1, ax1 = plt.subplots(1, 1, figsize=[9, 3], dpi=200)
fig2, ax2 = plt.subplots(1, 1, figsize=[5, 5], dpi=200)

night_names = ['Gewöhnungsnacht', 'Nacht 1', 'Nacht 2', 'Nacht 3']

hypno_pngs = []
dist_pngs = []
summaries = []

for n, (raw, hypno) in enumerate(zip(raws, hypnos)):
    basename = os.path.basename(os.path.splitext(raw.filenames[0])[0])
    try:
        n = int(basename[6])
    except ValueError:
        pass
    night = night_names[n]  # night id at position 6 of filename

    ax = ax1
    ax.clear()
    starttime = raw.info["meas_date"].replace(microsecond=0, tzinfo=None) # remove timezone and millisecond info
    plot_hypnogram(hypno, ax=ax, starttime=starttime,
                   title=f"Schlafprofil für '{night}', Start um {starttime}")

    # filter annotations that they are not overlapping in the plot
    # first split into blocks with 5 minute no annotation as separators
    # then take the first and the last annotation of that block.
    annotations = [annot for annot in raw.annotations if not 'New Segment' in annot['description']]

    tdiffs = np.diff([annot['onset'] for annot in annotations])
    annot_blocks_first = [annot for annot, tdiff in zip(annotations[1:], tdiffs) if tdiff>300]
    annot_blocks_last = [annot for annot, tdiff in zip(annotations[:-1], tdiffs) if tdiff>300]

    last_evening_marker = annot_blocks_last[0] if len(annot_blocks_last)>0 else {}
    first_morning_marker = annot_blocks_first[-1] if len(annot_blocks_first)>0 else {}
    
        

    lights_off_epoch = int(last_evening_marker.get('onset', 0)//30)
    lights_on_epoch = int(first_morning_marker.get('onset', -30 + len(raw)/raw.info['sfreq'])//30)

    if lights_off_epoch > np.argmax(hypno>0):
        # sometimes no annotations match, then the values are off
        lights_off_epoch = np.argmax(hypno>0)
        
    if lights_on_epoch < len(hypno)- np.argmax(hypno[::-1]>0): 
        # sometimes no annotations match, then the values are off
        lights_on_epoch = len(hypno)- np.argmax(hypno[::-1]>0)


    for i, annot in enumerate(annot_blocks_last + annot_blocks_first[-1:]):
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
    summary = sleep_utils.hypno_summary(hypno, lights_on_epoch=lights_on_epoch, lights_off_epoch=lights_off_epoch)
    summaries += [pd.Series(summary, name=night)]
    labels = ['Wach', 'S1', 'S2', 'SWS', 'REM']
    values = [summary['WASO'], summary['min_S1'], summary['min_S2'], summary['min_S3'], summary['min_REM']]
    percent = [summary['perc_W'], summary['perc_S1'], summary['perc_S2'], summary['perc_S3'], summary['perc_REM']]
    labelszip = zip(['\nWach', 'S1', 'S2', 'SWS', 'REM'], values, percent)
    if plt.rcParams['text.usetex'] :
        stageslabels = [f'\\textbf{{{l}}}'for l, m, p in labelszip]
    else:
        stageslabels = [f'{l}'for l, m, p in labelszip]

    autoptc = lambda x: f'{x:.1f} %\n{int(x*summary["TBT"]//100)} min'
    ax.pie(percent, labels=labels, explode=[0.025]*5, autopct=autoptc, textprops=dict(fontsize=16),
           labeldistance=1.1, pctdistance=0.7)
    ax.set_title(f"Verteilung '{night}'", fontsize=14, fontweight='bold')

    hypno_png = f'{report_dir}/hypno_{basename}.png'
    dist_png = f'{report_dir}/dist_{basename}.png'
    hypno_pngs += [hypno_png]
    dist_pngs += [dist_png]
    plt.pause(0.1)
    fig1.tight_layout(h_pad=3)
    fig2.tight_layout(h_pad=3)
    plt.pause(0.1)
    fig1.tight_layout(h_pad=3)
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
    'min_S2': '180-300',  # Minutes in stage S2
    'min_S3': '30-120',  # Minutes in stage S3
    'min_REM': '60-150',  # Minutes in REM sleep
    'sum_NREM': '210-420',  # Sum of Minutes in Non-REM
    'perc_W': '1-5%',  # Percentage of wake time
    'perc_S1': '2-8%',  # Percentage of stage S1 time
    'perc_S2': '40-60%',  # Percentage of stage S2 time
    'perc_S3': '10-25%',  # Percentage of stage S3 time
    'perc_REM': '15-30%',  # Percentage of REM sleep time
    'lat_S1': '5-25',  # Latency to stage S1 in minutes
    'lat_S2': '10-30',  # Latency to stage S2 in minutes
    'lat_S3': '20-40',  # Latency to stage S3 in minutes
    'lat_REM': '70-120',  # Latency to REM sleep in minutes
    'lights_off': 'N/A',  # Sleep onset after recording start in minutes
    'lights_on': 'N/A',  # Sleep offset after recording start
    'recording_length': 'N/A',  # Recording length in minutes (7 to 10 hours)
    'awakenings': '0-15',  # Number of awakenings
    'mean_dur_awakenings': '1-6',  # Mean duration of awakenings in minutes
    'FI': 'N/A',  # Fragmentation index
    'sleep_cycles': '3-5',  # Number of sleep cycles
    'stage_shifts': '10-70',  # Number of shifts of sleep stages
    'SQI': '0.7-1.0',  # Sleep quality index (ratio)
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
    'sum_NREM': 'Minuten im Non-REM (S2+S3)',
    'perc_W': 'Prozent der Wachzeit',
    'perc_S1': 'Prozent der Schlafphase S1',
    'perc_S2': 'Prozent der Schlafphase S2',
    'perc_S3': 'Prozent der Schlafphase S3',
    'perc_REM': 'Prozent der REM-Schlafzeit',
    'lat_S1': 'Latenz bis Schlafphase S1',
    'lat_S2': 'Latenz bis Schlafphase S2',
    'lat_S3': 'Latenz bis Schlafphase S3',
    'lat_REM': 'Latenz bis REM-Schlaf',
    'lights_off': 'Wann in etwa das Licht ausgemacht wurde',
    'lights_on': 'Wann in etwa das Licht angemacht wurde',
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
df_summaries.index.set_names('Kennwert', inplace=True)
#%% create MarkDown file

string = f"# Ihre Schlafdaten im Überblick\n"
string += """Im Folgenden finden Sie eine Übersicht über Ihre Schlafdaten. Die Analyse umfasst verschiedene Parameter, die wir näher erklären.

**Wichtige Hinweise**

Bitte beachten Sie, dass diese Analyse keine medizinische Untersuchung ist. Die dargestellten Daten können Fehler enthalten und die Richtwerte sind nur als grobe Orientierung zu verstehen. Es ist völlig normal, dass einzelne Nächte von diesen Durchschnittswerten abweichen. Die Durchschnittswerten basieren auf Mittelwerten von Analysen über viele Nächte einer grossen Probandengruppe. Die Schlafaufnahme ist für wissenschaftliche und nicht für diagnostische Zwecke optimiert.


### Hypnogramm und Schlafphasen:

Hypnogramme (oben): Diese Diagramme zeigen Ihre Schlafstadien über drei verschiedene Nächte:
- **Gewöhnungsnacht:** Diese Nacht diente dazu, sich an die Schlafumgebung zu gewöhnen.
- **Nacht 1:** Diese Nacht wurde vor der Rauchentwöhnung aufgezeichnet.
- **Nacht 2:** Diese Nacht wurde nach der Rauchentwöhnung aufgezeichnet.

Tortendiagramme (unten): Diese Diagramme stellen die prozentuale Verteilung der verschiedenen Schlafphasen dar:

- **W:** Wachzustand
- **REM:** REM-Schlaf (Rapid Eye Movement)
- **S1:** Schlafstadium 1 (Einschlafphase)
- **S2:** Schlafstadium 2 (leichter Schlaf)
- **SWS:** Tiefschlaf (Slow-Wave Sleep, auch S3 genannt)

"""

for i, hypno_png in enumerate(hypno_pngs):
    string += f'<img src="./{os.path.basename(hypno_png)}" alt="hypno_{i}" height="220px"/><br><br>'
    # string += f'![hypnogram_{i}](./{os.path.basename(hypno_png)})\n\n'


for i, dist_png in enumerate(dist_pngs):
    string += f'<img src="./{os.path.basename(dist_png)}" alt="dist_{i}" width="{1/(len(dist_pngs)+1)*100}%"/>'

string += '<div style="page-break-after: always;"></div><br>'

string += f"\n\n\n\n## Tabellarische Werte\n{df_summaries.to_markdown(stralign='right')}\n"
string += """

**Erklärungen zu den Schlafphasen**

- **S1 (Einschlafphase):** Dies ist die leichteste Schlafphase, die als Übergang zwischen Wachsein und Schlafen dient. In dieser Phase können leichte Bewegungen und schnelle Wechsel zurück ins Wachsein vorkommen.
- **S2 (Leichtschlaf):** Dies ist eine Phase des leichten Schlafs, die den Großteil des Schlafs ausmacht. In dieser Phase verlangsamt sich die Herzfrequenz und die Körpertemperatur sinkt. Es ist schwieriger, aus dieser Phase aufzuwachen als aus S1.
- **S3 (Tiefschlaf):** Diese Phase ist auch als Tiefschlaf oder Slow-Wave Sleep (SWS) bekannt. Sie ist für die körperliche Erholung und das Immunsystem besonders wichtig. Aufwachen aus dieser Phase kann zu Desorientierung führen.
- **REM (Rapid Eye Movement) Schlaf:** Diese Phase ist durch schnelle Augenbewegungen gekennzeichnet und wird oft mit intensivem Träumen in Verbindung gebracht. 



Nochmals vielen Dank für Ihre Teilnahme und Ihr Vertrauen!

Ihr Studienteam"""


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
html_text = markdown2.markdown(string.replace('./', report_dir),  extras=['tables', 'fenced-code-blocks', 'cuddled-lists'])
# Add inline CSS for better table formatting
html_text = f"""
    <html>
    <head>
<meta charset="UTF-8">
<style>
body {{
    font-family: Arial, sans-serif;
    font-size: 13px; /* Decreased global font size by 1 point */
}}
table {{
    width: 70%;
    border-collapse: collapse;
}}
th, td {{
    padding: 5px 6px ; /* Add more space between columns */
    text-align: left;
    font-size: 12px; /* Decreased global font size by 1 point */
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
config = pdfkit.configuration(wkhtmltopdf='C:/Users/simon.kern/Nextcloud/Projects/sleep-utils/sleep_utils/examples/bin/wkhtmltopdf.exe')

assert pdfkit.from_string(html_text, file_pdf, configuration=config, options={"enable-local-file-access": ""}), 'converting failed'

print(f'PDF saved to {file_pdf}')

subprocess.call(['open', file_pdf])

input('Press enter to exit...')
