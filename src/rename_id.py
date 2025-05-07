import os
import re
import shutil

# Paths
source_folder = './Prolific/data_orig'
destination_folder = './Prolific/data_all'

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Mapping from original participant IDs to new IDs (p1, p2, ...)
id_map = {}
id_counter = 1

# List all .txt files
all_files = [f for f in os.listdir(source_folder) if f.endswith('.txt')]

# Function to extract the longest alphanumeric token from filename
def extract_id(filename):
    tokens = re.split(r'[_\.]', filename)
    longest = max(tokens, key=len)
    return longest

# First pass: create ID mapping
for filename in all_files:
    participant_id = extract_id(filename)
    if participant_id not in id_map:
        id_map[participant_id] = f'p{id_counter}'
        id_counter += 1

# Second pass: process files
for filename in all_files:
    participant_id = extract_id(filename)
    new_id = id_map[participant_id]
    
    # Read and replace contents
    src_path = os.path.join(source_folder, filename)
    with open(src_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace mTurkID value
    content = content.replace(participant_id, new_id)
    
    # Rename the file accordingly
    new_filename = filename.replace(participant_id, new_id)
    dst_path = os.path.join(destination_folder, new_filename)
    
    # Write updated content to new file
    with open(dst_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f'Processed: {filename} -> {new_filename}')
