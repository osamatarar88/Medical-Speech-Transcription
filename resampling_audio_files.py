import os
import librosa
import soundfile as sf

# Define your directories
TRAIN_DIR = 'recordings/train'
VALIDATE_DIR = 'recordings/validate'
TEST_DIR = 'recordings/test'
TARGET_SAMPLE_RATE = 16000  # Define the target sample rate (e.g., 16 kHz)

# Function to list all audio files in a directory
def list_audio_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')]

# Function to resample audio files to the target sample rate
def resample_audio(file_path, target_sr):
    try:
        # Load the audio file with its original sample rate
        y, sr = librosa.load(file_path, sr=None)
        
        if sr != target_sr:
            print(f"Resampling {file_path} from {sr} Hz to {target_sr} Hz")
            # Resample the audio
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            # Save the resampled audio back to the same file
            sf.write(file_path, y_resampled, target_sr)
        else:
            print(f"{file_path} already has the target sample rate of {target_sr} Hz")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Collect all audio files from the directories
train_files = list_audio_files(TRAIN_DIR)
validate_files = list_audio_files(VALIDATE_DIR)
test_files = list_audio_files(TEST_DIR)

# Resample all audio files
print("Resampling train files...")
for file in train_files:
    resample_audio(file, TARGET_SAMPLE_RATE)

print("\nResampling validation files...")
for file in validate_files:
    resample_audio(file, TARGET_SAMPLE_RATE)

print("\nResampling test files...")
for file in test_files:
    resample_audio(file, TARGET_SAMPLE_RATE)

print("\nResampling completed.")
