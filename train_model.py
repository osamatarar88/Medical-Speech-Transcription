import os
import json
import torch
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecCTCModel
from pytorch_lightning import Trainer

# Function to extract labels from manifest files
def extract_labels_from_manifest(manifest_paths):
    labels_set = set()
    for manifest_path in manifest_paths:
        with open(manifest_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                text = entry.get('text', '')  # Extract the 'text' field
                labels_set.update(text)  # Collect all unique characters

    labels = sorted(list(labels_set))
    labels.insert(0, "_")  # Add CTC blank token at index 0
    return labels

# Manifest paths
TRAIN_MANIFEST = 'train_manifest.json'
VAL_MANIFEST = 'val_manifest.json'
TEST_MANIFEST = 'test_manifest.json'

# Extract labels from manifest files
manifest_paths = [TRAIN_MANIFEST, VAL_MANIFEST, TEST_MANIFEST]
LABELS = extract_labels_from_manifest(manifest_paths)
print("Extracted Labels:", LABELS)

# Load the pre-trained ASR model (QuartzNet)
asr_model = EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")

# Change the model vocabulary to match extracted labels
asr_model.change_vocabulary(new_vocabulary=LABELS)

# Set up data loaders with sample rate and extracted labels
asr_model.setup_training_data(
    train_data_config={
        'manifest_filepath': TRAIN_MANIFEST,
        'sample_rate': 16000,
        'batch_size': 4,
        'labels': LABELS,
        'shuffle': True,
        'num_workers': 2 
    }
)

asr_model.setup_validation_data(
    val_data_config={
        'manifest_filepath': VAL_MANIFEST,
        'sample_rate': 16000,
        'batch_size': 4,
        'labels': LABELS,
        'num_workers': 2 
    }
)

asr_model.setup_test_data(
    test_data_config={
        'manifest_filepath': TEST_MANIFEST,
        'sample_rate': 16000,
        'batch_size': 4,
        'labels': LABELS,
        'num_workers': 2 
    }
)

# Determine the accelerator type and number of devices
accelerator = "gpu" if torch.cuda.is_available() else "cpu"
devices = 1 if torch.cuda.is_available() else torch.cuda.device_count() or 1

# Initialize the PyTorch Lightning Trainer
trainer = Trainer(
    max_epochs=20,  # Set the number of epochs
    accelerator=accelerator,
    devices=devices,
    log_every_n_steps=10,  # Log every 10 steps to monitor training
)

# Train the model
trainer.fit(asr_model)

# Test the model after training (Optional)
#trainer.test(asr_model)
