# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 15:10:42 2025

@author: Simon
"""
import os
import json
import appdirs
import tkinter as tk
from tkinter import Listbox, Scrollbar, Label, Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
from natsort import natsorted
from tkinter import filedialog
from pathlib import Path
from natsort import natsort_key


def config_load():
    config_dir = appdirs.user_config_dir('sleep-utils')
    config_file = os.path.join(config_dir, 'last_used.json')
    os.makedirs(config_dir, exist_ok=True)

    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        config = {}
    return config

def config_save(config):
    config_dir = appdirs.user_config_dir('sleep-utils')
    config_file = os.path.join(config_dir, 'last_used.json')
    os.makedirs(config_dir, exist_ok=True)

    with open(config_file, 'w') as f:
        config = json.dump(config, f)


def select_files(default_dir=None, title='Select file(s)',
                 exts=['*.fif', '*.vhdr', '*.edf']):
    """A GUI that lets you select several files from possible subfolders

    Opens a Tkinter GUI that allows the user to select multiple files or folders,
    shows the selections in a multiselect listbox, and returns the final list
    of selected files when the user clicks "Proceed with these files".

    Additionally, below the "Add all files in folder" button there is a button
    called "Load previous selection" that loads the previous file selection stored
    in the config json, and when "Proceed with these files" is clicked, the selection
    is saved to the config.

    :param default_dir: Default directory to open in file/folder dialogs (optional)
    :param title: Title for the "add file(s)" dialog
    :param exts: Default extensions for filtering. E.g., ['*.fif', '*.vhdr', '*.edf'].
    :return: A list of paths selected by the user or None if the user cancels.
    """
    root = tk.Tk()
    root.title("Select Multiple Files")

    # Configure grid: we now need 7 rows (0-6) and 4 columns (0-3)
    for i in range(7):
        root.grid_rowconfigure(i, weight=1)
    # Make sure columns 0..3 expand
    for i in range(4):
        root.grid_columnconfigure(i, weight=1)

    selected_files = []

    # Frame for the listbox+scrollbar (occupies rows 0..5, columns 0..2)
    list_frame = tk.Frame(root)
    list_frame.grid(row=0, column=0, rowspan=6, columnspan=3, padx=5, pady=5, sticky="nsew")

    # Scrollbar for the listbox
    scrollbar = tk.Scrollbar(list_frame, orient="vertical")
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Listbox (multiselect) with scrollbar
    listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, width=60, yscrollcommand=scrollbar.set)
    listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.config(command=listbox.yview)

    # ----- RIGHT COLUMN WIDGETS (column 3) -----

    # Row 0: "Filter:" label
    tk.Label(root, text="Filter:").grid(row=0, column=3, sticky="nw", padx=5, pady=5)
    # Row 1: Filter text entry
    filter_var = tk.StringVar(value=', '.join(exts))
    filter_entry = tk.Entry(root, textvariable=filter_var, width=20)
    filter_entry.grid(row=1, column=3, padx=5, pady=5, sticky="nw")

    # Row 2: "Add all files in folder" button
    def add_folder():
        config = config_load()
        default_dir_cfg = config.get('select_files.last_directory_addfolder', default_dir)

        folder_path = filedialog.askdirectory(title="Select Folder", initialdir=default_dir_cfg)
        if not folder_path:
            return

        config['select_files.last_directory_addfolder'] = folder_path
        config_save(config)

        # Re-check exts in case filter was changed
        _filters = [e.strip() for e in filter_var.get().split(',')]
        files_found = list_files(folder_path, exts=_filters, recursive=True)
        if files_found:
            for file in files_found:
                if file in selected_files:
                    continue
                listbox.insert(tk.END, file)
                selected_files.append(file)
        else:
            print(f'No files found that match {_filters} in "{folder_path}" and subfolders')

    add_folder_btn = tk.Button(root, text="Add all files in folder", command=add_folder)
    add_folder_btn.grid(row=2, column=3, padx=5, pady=5, sticky="nw")

    # Row 3: "Load previous selection" button
    def load_previous_selection():
        config = config_load()
        prev_selection = config.get('select_files.previous_selection')
        if prev_selection:
            for file in prev_selection:
                if file in selected_files:
                    continue
                listbox.insert(tk.END, file)
                selected_files.append(file)
        else:
            print("No previous selection found in config.")

    load_prev_btn = tk.Button(root, text="Load previous selection", command=load_previous_selection)
    load_prev_btn.grid(row=3, column=3, padx=5, pady=5, sticky="nw")

    # Row 4: "Add file(s)" button
    def add_files():
        _filters = [e.strip() for e in filter_var.get().split(',')]
        _patterns = " ".join(_filters)
        _filetypes = [("Allowed files", _patterns)]
        config = config_load()
        default_dir_cfg = config.get('select_files.last_directory_addfolder', default_dir)

        file_paths = filedialog.askopenfilenames(filetypes=_filetypes, title=title, initialdir=default_dir_cfg)
        if file_paths:
            config['select_files.last_directory_addfolder'] = os.path.dirname(file_paths[0])
            config_save(config)

        for file in file_paths:
            if file in selected_files:
                continue
            listbox.insert(tk.END, file)
            selected_files.append(file)

    add_files_btn = tk.Button(root, text="Add file(s)", command=add_files)
    add_files_btn.grid(row=4, column=3, padx=5, pady=5, sticky="nw")

    # Row 5: "Remove selected" button
    def remove_selected():
        selected_indices = list(listbox.curselection())
        for i in reversed(selected_indices):
            listbox.delete(i)
            del selected_files[i]

    remove_btn = tk.Button(root, text="Remove selected", command=remove_selected)
    remove_btn.grid(row=5, column=3, padx=5, pady=5, sticky="nw")

    # ----- BOTTOM ACTION BUTTONS (occupying row 6 in columns 0-2) -----

    # Row 6: "Proceed with these files" and "Cancel"
    def on_proceed():
        # Save the current selection to config
        config = config_load()
        config['select_files.previous_selection'] = selected_files
        config_save(config)
        root.quit()  # Exit main loop

    proceed_btn = tk.Button(root, text="Proceed with these files", command=on_proceed)
    proceed_btn.grid(row=6, column=0, padx=5, pady=10, sticky="nsew")

    def on_cancel():
        nonlocal selected_files
        selected_files = None
        root.quit()  # Exit main loop

    cancel_btn = tk.Button(root, text="Cancel", command=on_cancel)
    cancel_btn.grid(row=6, column=1, columnspan=2, padx=5, pady=10, sticky="nsew")

    root.mainloop()
    root.destroy()

    return selected_files

def choose_file(default_dir=None, default_file=None, exts='txt',
                title='Choose file', mode='open', multiple=False):
    """
    Open a file chooser dialoge with tkinter.

    :param default_dir: Where to open the dir, if set to None, will start at wdir
    :param exts: A string or list of strings with extensions etc: 'txt' or ['txt','csv']
    :returns: the chosen file
    """
    config = config_load()

    if default_dir is None:
        default_dir = config.get('last_directory')

    root = Tk()
    root.deiconify()
    root.lift()
    root.attributes('-topmost', True)
    root.attributes('-topmost', False)
    root.update()
    if isinstance(exts, str): exts = [exts]
    if mode=='open':
       name = askopenfilename(initialdir=default_dir,
                              default_file=default_file,
                              parent=root,
                              multiple=multiple,
                              title = title,
                              filetypes =(*[("File", "*.{}".format(ext)) for ext in exts],
                                           ("All Files","*.*")))
       if multiple:
           assert not isinstance(name, str)
           name = natsorted(name)
    elif mode=='save':
        assert not multiple, 'multiple must be false for saving'
        name = asksaveasfilename(initialdir=default_dir,
                              initialfile=default_file,
                              parent=root,
                              title = title,
                              filetypes =(*[("File", "*.{}".format(ext)) for ext in exts],
                                         ("All Files","*.*")))
        if not name.endswith(exts[0]):
            name += f'.{exts[0]}'
    else:
        raise Exception(f'unknown mode: {mode}')
    root.update()
    root.destroy()

    config['last_directory'] = os.path.dirname(name[0] if multiple else name)
    config_save(config)

    if not name:
        print("ERROS: No file chosen")
    else:
        return name



def display_textbox(title='Enter text', label='Please enter text', text=''):
    window = tk.Tk()
    window.deiconify()
    window.lift()
    window.attributes('-topmost', True)
    window.attributes('-topmost', False)
    window.title()

    # Label
    label = tk.Label(window, text=label)
    label.pack()

    # Text input field with line wrapping
    text_input = tk.Text(window, wrap=tk.WORD, height=5, width=50,)
    text_input.configure(font=("Arial", 8))  # Reducing font size

    text_input.pack(pady=5, padx=5)

    if text:
        text_input.insert(tk.END, text)

    x = ['']
    # OK button
    def on_ok():
        # Get the text from the input field
        text = text_input.get("1.0", tk.END).strip()
        x[0] = text
        window.destroy()  # Close the window

    ok_button = tk.Button(window, text="OK", command=on_ok)
    ok_button.pack()

    # Run the Tkinter event loop
    window.mainloop()
    return x[0]


def display_listbox(lists, selected=[], labels=[], title='select items',
                    mode=tk.MULTIPLE):
    """
    Displays a tkinter window with multiple listboxes for item selection.

    Parameters
    ----------
    lists : list of lists
        A list where each element is a list of items to be displayed in a
        separate listbox. Each listbox corresponds to one list of items.
    selected : list of lists, optional
        A list where each element is a list of preselected items for the
        corresponding listbox. The default is an empty list, meaning no
        items are preselected.
    labels : list of str, optional
        A list of labels to be displayed above each listbox. If fewer labels
        are provided than the number of listboxes, the remaining listboxes
        will have no label. The default is an empty list.
    title : str, optional
        The title of the tkinter window. The default is 'select items'.
    mode : tkinter constant, optional
        The selection mode for the listboxes. Can be `tk.SINGLE` for single
        selection or `tk.MULTIPLE` for multiple selection. alternatively
        just 'single' or 'multiple'. The default is `tk.MULTIPLE`.

    Returns
    -------
    selected_items : list of lists
        A list where each element is a list of items selected from the
        corresponding listbox. If no items are selected in a listbox, the
        corresponding list will be empty.
    """
    # Create main tkinter window
    root = Tk()
    root.deiconify()
    root.lift()
    root.attributes('-topmost', True)
    root.attributes('-topmost', False)
    root.title(title)

    assert len(selected) <= len(lists), 'more selected than lists'
    # assert len(labels) <= len(lists), 'more labels than lists'
    assert isinstance(lists, (list, tuple)), 'lists must be list of lists'
    for items in lists:
        assert isinstance(items, (list, tuple)), 'items must be list of lists'

    num_listboxes = len(lists)

    # Create labels and listboxes
    lbls = []
    listboxes = []

    for i in range(num_listboxes):
        items = lists[i]
        preselected = selected[i] if i < len(selected) else []
        text = labels[i] if i < len(labels) else ''

        # Create label above the listbox
        label = Label(root, text=text)
        label.grid(row=0, column=i, padx=5, pady=5)
        lbls.append(label)

        # Create listbox
        listbox = Listbox(root, selectmode=mode, exportselection=False)
        listbox.grid(row=1, column=i, padx=5, pady=5)
        listboxes.append(listbox)

        for item in items:
            listbox.insert(tk.END, str(item))
            if item in preselected:
                listbox.selection_set(items.index(item))  # Select the item

        # Create scrollbar for each listbox
        scrollbar = Scrollbar(root, orient=tk.VERTICAL, command=listbox.yview)
        scrollbar.grid(row=1, column=i, sticky=tk.NS + tk.E)
        listbox.config(yscrollcommand=scrollbar.set)

    # Create OK button
    ok_button = tk.Button(root, text="OK", command=root.quit)
    ok_button.grid(row=2, columnspan=num_listboxes, pady=10)

    # Run the tkinter event loop
    root.mainloop()

    # Gather selected items before closing
    selected_items = []
    for i in range(num_listboxes):
        selected_item_list = listboxes[i].curselection()
        selected_items.append([listboxes[i].get(idx) for idx in selected_item_list])

    root.destroy()
    return selected_items


def list_files(path, exts=None, patterns=None, relative=False, recursive=False,
               subfolders=None, only_folders=False, max_results=None,
               case_sensitive=False):
    """
    will make a list of all files with extention exts (list)
    found in the path and possibly all subfolders and return
    a list of all files matching this pattern

    :param path:  location to find the files
    :type  path:  str
    :param exts:  extension of the files (e.g. .jpg, .jpg or .png, png)
                  Will be turned into a pattern internally
    :type  exts:  list or str
    :param pattern: A pattern that is supported by pathlib.Path,
                  e.g. '*.txt', '**\rfc_*.clf'
    :type:        str
    :param fullpath:  give the filenames with path
    :type  fullpath:  bool
    :param subfolders
    :param return_strings: return strings, else returns Path objects
    :return:      list of file names
    :type:        list of str
    """
    def insensitive_glob(pattern):
        f = lambda c: '[%s%s]' % (c.lower(), c.upper()) if c.isalpha() else c
        return ''.join(map(f, pattern))

    if subfolders is not None:
        import warnings
        warnings.warn("`subfolders` is deprecated, use `recursive=` instead", DeprecationWarning)
        recursive = subfolders

    if isinstance(exts, str): exts = [exts]
    if isinstance(patterns, str): patterns = [patterns]
    assert isinstance(path, str), "path needs to be a str"
    assert os.path.exists(path), 'Path {} does not exist'.format(path)
    if patterns is None: patterns = []
    if exts is None: exts = []

    if patterns==[] and exts == []:
        patterns = ['*']

    for ext in exts:
        ext = ext.replace('*', '')
        pattern = '*' + ext
        patterns.append(pattern.lower())

    # if recursiveness is asked, prepend the double asterix to each pattern
    if recursive: patterns = ['**/' + pattern for pattern in patterns]

    # collect files for each pattern
    files = []
    fcount = 0
    for pattern in patterns:
        if not case_sensitive:
            pattern = insensitive_glob(pattern)
        for filename in Path(path).glob(pattern):
            if filename.is_file() and filename not in files:
                files.append(filename)
                fcount += 1
                if max_results is not None and max_results<=fcount:
                    break

    # turn path into relative or absolute paths
    files = [file.relative_to(path) if relative else file.absolute() for file in files]
    files = [os.path.join(file) for file in files]

    files = set(files)  # filter duplicates
    # by default: return strings instead of Path objects
    return sorted(files, key=natsort_key)


# You can test the function as follows:
if __name__ == "__main__":
    result = select_files()
    print("User selected:", result)
