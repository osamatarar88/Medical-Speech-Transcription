
# Medical Speech Transcription using NVIDIA NeMo

This repository implements an **Automatic Speech Recognition (ASR)** system on a medical speech dataset using the **NVIDIA NeMo ASR toolkit**. It processes medical audio recordings, trains a model for transcription, and provides an interface for real-time ASR through a **Streamlit GUI**.

## Dataset
The dataset used for this project is the **Medical Speech, Transcription, and Intent Dataset** available on Kaggle:  
[Kaggle Dataset Link](https://www.kaggle.com/datasets/paultimothymooney/medical-speech-transcription-and-intent)

- The dataset contains audio recordings and corresponding transcription files, which are utilized for training and validating the ASR model.

The **`overview-of-recordings.csv`** file provides a summary of the audio files in the dataset, including details such as speaker intent, transcription, and file paths.

## Project Workflow

### 1. Preparing the Dataset
- **File:** `resampling_audio_files.py`  
  - Ensures that all audio files are **resampled to 16kHz**, as different audio files in the original dataset may have inconsistent sampling rates.  
  - This step guarantees uniformity in audio processing.

- **File:** `inspect_sampling_rate.py`  
  This script verifies the sampling rate of all audio files to ensure they meet the **16kHz** requirement.

### 2. Creating Manifests for NeMo
- **File:** `create_manifests.py`  
  - This script generates **JSON manifest files** required by NeMo for training, validation, and testing.
  - **Outputs:**
    - `train_manifest.json`
    - `val_manifest.json`
    - `test_manifest.json`

Each manifest contains the audio file path, transcription, and metadata for each sample, formatted specifically for the NeMo ASR pipeline.

### 3. Model Training and Evaluation
- **File:** `training_model.ipynb`  
  This notebook contains the entire workflow for training and evaluating the **QuartzNet model**, a popular convolutional ASR architecture:

  - **Steps in the notebook:**
    1. Loading the resampled data and manifests.
    2. Configuring the QuartzNet architecture for medical ASR.
    3. Training the model using the training dataset (`train_manifest.json`).
    4. Validating the model with `val_manifest.json`.
    5. Evaluating the model on the test set (`test_manifest.json`).

- **Model Output:**  
  The trained model is saved as **`quartznet_model.nemo`**, which can be directly loaded for inference.

## Deploying the Model with Streamlit GUI
- **File:** `streamlit_app.py`  
  This file contains the **Streamlit GUI** implementation, providing an easy-to-use interface for users to upload an audio file and get real-time transcription. 

  **Usage:**
  1. Run the Streamlit app:  
     ```bash
     streamlit run streamlit_app.py
     ```
  2. Upload an audio file through the interface.
  3. View the transcription generated by the trained QuartzNet model.

## Project Structure
```
Medical-Speech-Transcription/
│
├── create_manifests.py        # Script to create NeMo-compatible manifests
├── inspect_sampling_rate.py   # Inspects sampling rate of audio files
├── overview-of-recordings.csv # Overview of the dataset content
├── quartznet_model.nemo       # Trained QuartzNet model
├── requirements.txt           # Python dependencies for the project
├── resampling_audio_files.py  # Resamples audio files to 16kHz
├── streamlit_app.py           # Streamlit GUI for real-time ASR
├── test_manifest.json         # Test manifest for NeMo
├── train_manifest.json        # Training manifest for NeMo
├── training_model.ipynb       # Model training and evaluation notebook
├── val_manifest.json          # Validation manifest for NeMo

```

## Setup and Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd Medical-Speech-Transcription
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure that all audio files are resampled to **16kHz** using:
   ```bash
   python resampling_audio_files.py
   ```

4. Create the required manifests:
   ```bash
   python create_manifests.py
   ```

## Usage
### Training the Model
- Use the provided Jupyter notebook `training_model.ipynb` for training the QuartzNet model on the medical dataset.

### Running the Streamlit App
- Launch the Streamlit interface with:
  ```bash
  streamlit run streamlit_app.py
  ```

## Model and Toolkit
- **Model Architecture:** QuartzNet (Convolutional ASR)
- **ASR Toolkit:** NVIDIA NeMo

QuartzNet is known for its efficiency and accuracy in speech recognition tasks, making it well-suited for medical transcription.

## Contributing
Feel free to open issues or submit pull requests for improvements. Contributions are welcome!

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- NVIDIA for the **NeMo toolkit**  
- **Kaggle** for the medical dataset  
- The creators of the **Medical Speech, Transcription, and Intent Dataset**
