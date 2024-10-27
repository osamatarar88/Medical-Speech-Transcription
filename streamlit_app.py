import streamlit as st
import nemo.collections.asr as nemo_asr
import soundfile as sf

# Load the trained QuartzNet model
@st.cache_resource  # Cache the model to avoid reloading each time
def load_model():
    model = nemo_asr.models.EncDecCTCModel.restore_from("quartznet_model.nemo")
    return model

def predict_asr(audio_path, model):
    # Use the transcribe method to get the ASR output
    transcription = model.transcribe(paths2audio_files=[audio_path])
    return transcription[0]  # Extract the first result

# Streamlit UI
st.title("ASR - Automatic Speech Recognition")
st.write("Upload a .wav file to get the ASR output.")

uploaded_file = st.file_uploader("Upload Audio File", type=["wav"])

if uploaded_file is not None:
    # Display the uploaded audio
    st.audio(uploaded_file, format='audio/wav')

    # Save the uploaded file locally for processing
    audio_path = f"temp_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load the model
    model = load_model()

    # Make prediction
    st.write("Processing...")
    transcription = predict_asr(audio_path, model)

    # Display the transcription
    st.write("### ASR Output:")
    st.write(transcription)
