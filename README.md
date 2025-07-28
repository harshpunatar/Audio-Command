# 🎙️ Audio Command Recognition System

A deep learning-powered system that recognizes spoken commands from `.wav` audio files. Built with a Convolutional Neural Network (CNN) trained on the Google Speech Commands dataset.

## 🧠 Model

- Model file: `model.h5`
- Input Features: MFCC + Mel Spectrogram
- Accuracy: **84.2%**
- Framework: TensorFlow / Keras

## 🚀 Streamlit App

The `app.py` file provides an interactive Streamlit interface to upload an audio file and predict the spoken command.

### 🔧 To Run Locally

1. Install dependencies:

   ```bash
   pip install streamlit librosa tensorflow numpy
   ```
  
2. Run the App:
   ```bash
   streamlit run app.py
   ```

3. Upload a .wav file (e.g., "yes.wav", "no.wav") to test the model.


###📌 Future Scope
Add support for more commands

Improve accuracy with larger architecture

Real-time voice input via microphone

Add data visualization in the UI


Created with ❤️ by Harsh Punatar
