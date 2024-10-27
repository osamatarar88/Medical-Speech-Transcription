import os
import librosa

# Define your directories
TRAIN_DIR = 'recordings/train'
VALIDATE_DIR = 'recordings/validate'
TEST_DIR = 'recordings/test'

# Function to list all audio files in a directory
def list_audio_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')]

# Function to print the sample rate of audio files
def check_sample_rate(audio_files):
    for file in audio_files:
        try:
            # Load the audio file
            y, sr = librosa.load(file, sr=None)  # sr=None to get the original sample rate
            print(f"Sample rate of {file}: {sr} Hz")
        except Exception as e:
            print(f"Error loading {file}: {e}")

# Collect all audio files from your directories
train_files = list_audio_files(TRAIN_DIR)
validate_files = list_audio_files(VALIDATE_DIR)
test_files = list_audio_files(TEST_DIR)

# Check sample rate of each audio file
print("Checking sample rates in train files...")
check_sample_rate(train_files)

print("\nChecking sample rates in validation files...")
check_sample_rate(validate_files)

print("\nChecking sample rates in test files...")
check_sample_rate(test_files)
