import os
import shutil
from pathlib import Path

def merge_folders(root_input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    input_folders = [
        os.path.join(root_input_folder, name)
        for name in sorted(os.listdir(root_input_folder))
        if os.path.isdir(os.path.join(root_input_folder, name))
    ]
    
    all_files = []
    
    # Collect all .pt files from input folders
    for folder in input_folders:
        files = sorted(Path(folder).glob('ex_*.pt'))  # assumes ex_??.pt format
        all_files.extend(files)
    
    print(f"Found {len(all_files)} total files.")
 
    # Copy and rename each file
    for i, file_path in enumerate(all_files, start=1):
        new_filename = f"ex_{i}.pt"
        destination = Path(output_folder) / new_filename
        shutil.copy(file_path, destination)
    
    print(f"All files merged into: {output_folder}")

root_input_folder = '/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment_full_anm/mic_train_ds_preprocessed'
output_folder = '/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment_full_anm/mic_train_ds_preprocessed_merged'

merge_folders(root_input_folder, output_folder)

#runai-cmd --name merge2  -g 0.1 --cpu-limit 20 -- "conda activate venv && python /Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/utils/merge_folders.py"
#runai-bgu submit python -n merge -c 20 -m 40G -g 0.2 --conda venv -- "python /Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/utils/merge_folders.py"
