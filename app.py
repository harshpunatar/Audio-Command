import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import tempfile

# Load model
model = tf.keras.models.load_model("audio_command_model.h5")

# Constants
SAMPLE_RATE = 16000
DURATION = 1
SAMPLES_PER_FILE = SAMPLE_RATE * DURATION
commands = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

# Feature extraction
def extract_features(signal):
    if len(signal) < SAMPLES_PER_FILE:
        signal = np.pad(signal, (0, SAMPLES_PER_FILE - len(signal)))
    else:
        signal = signal[:SAMPLES_PER_FILE]

    mfcc = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_mfcc=40)
    mel = librosa.feature.melspectrogram(y=signal, sr=SAMPLE_RATE, n_mels=40)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    combined = np.concatenate((mfcc, mel_db), axis=0)
    return combined.T

# Streamlit UI
st.title("🎙️ Audio Command Recognition")
uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    signal, _ = librosa.load(tmp_path, sr=SAMPLE_RATE)
    features = extract_features(signal)
    features = np.expand_dims(features, axis=0)

    prediction = model.predict(features)
    predicted_label = commands[np.argmax(prediction)]

    st.success(f"Predicted Command: **{predicted_label}**")
