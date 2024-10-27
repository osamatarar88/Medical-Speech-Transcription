import os
import pandas as pd
import librosa
import json  # Import json to serialize entries

# Define the paths to the recordings folders
TRAIN_DIR = 'recordings/train'
VALIDATE_DIR = 'recordings/validate'
TEST_DIR = 'recordings/test'

# Load the metadata CSV
metadata = pd.read_csv('overview-of-recordings.csv')

# Function to get the correct audio path for a given file name
def get_audio_path(file_name):
    # Possible paths for each file
    possible_paths = [
        os.path.join(TRAIN_DIR, file_name),
        os.path.join(VALIDATE_DIR, file_name),
        os.path.join(TEST_DIR, file_name)
    ]

    # Check each possible path
    for path in possible_paths:
        if os.path.exists(path):
            return path

    # If file not found, raise an error
    raise FileNotFoundError(f"Audio file not found: {file_name}")

# Function to create a NeMo-compatible manifest
def create_manifest(df, manifest_path):
    with open(manifest_path, 'w') as f:
        for _, row in df.iterrows():
            # Get the full path to the audio file
            audio_path = get_audio_path(row['file_name'])

            # Create the entry for the manifest
            entry = {
                "audio_filepath": audio_path,
                "text": row['phrase'],  # Assuming 'phrase' contains the transcription
                "duration": librosa.get_duration(filename=audio_path)
            }

            # Write the entry as a JSON object, ensuring double quotes
            json_entry = json.dumps(entry)
            f.write(json_entry + '\n')

# Split the dataset into training, validation, and test sets
train_files = metadata.sample(frac=0.8, random_state=42)
val_test_files = metadata.drop(train_files.index)
val_files = val_test_files.sample(frac=0.5, random_state=42)
test_files = val_test_files.drop(val_files.index)

# Create the manifest files
print("Creating train manifest...")
create_manifest(train_files, 'train_manifest.json')

print("Creating validation manifest...")
create_manifest(val_files, 'val_manifest.json')

print("Creating test manifest...")
create_manifest(test_files, 'test_manifest.json')

print("Manifests created successfully.")
