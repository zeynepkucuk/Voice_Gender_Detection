
import streamlit as st
import pickle
import numpy as np
import os
import librosa
# Updated import to avoid deprecation warning
from sklearn.mixture import GaussianMixture  # Direct import from sklearn.mixture
import tempfile
import matplotlib.pyplot as plt
import time

# Set page configuration
st.set_page_config(
    page_title="Voice Gender Classifier",
    page_icon="🎤",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .result-text {
        font-size: 1.8rem;
        font-weight: bold;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .male-result {
        background-color: #bbdefb;
        color: #0d47a1;
    }
    .female-result {
        background-color: #f8bbd0;
        color: #880e4f;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Function to load models
@st.cache(allow_output_mutation=True)
def load_models():
    """Load gender classification models and return them"""
    try:
        with open('male.gmm', 'rb') as male_file:
            gmm_male = pickle.load(male_file)
        with open('female.gmm', 'rb') as female_file:
            gmm_female = pickle.load(female_file)
        return gmm_male, gmm_female
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Function to extract features from audio
def extract_features(audio_path, sr=16000):
    """Extract MFCC features from audio file"""
    try:
        # Load audio file with librosa
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
        
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Transpose to get time as first dimension
        mfcc = mfcc.T
        
        return mfcc
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

# Function to predict gender
def predict_gender(audio_path, gmm_male, gmm_female):
    """Predict gender from audio file"""
    try:
        # Extract features
        features = extract_features(audio_path)
        
        if features is None:
            return None, None, None
        
        # Compute log likelihood scores
        male_score = gmm_male.score(features)
        female_score = gmm_female.score(features)
        
        # Determine gender based on higher score
        if male_score > female_score:
            return "Male", male_score, female_score
        else:
            return "Female", male_score, female_score
    except Exception as e:
        st.error(f"Error predicting gender: {e}")
        return None, None, None

# Function to display audio waveform
def plot_waveform(audio_path):
    """Plot audio waveform"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        fig, ax = plt.subplots(figsize=(10, 2))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title('Audio Waveform')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        return fig
    except Exception as e:
        st.error(f"Error plotting waveform: {e}")
        return None

# Main app
def main():
    # Load models
    gmm_male, gmm_female = load_models()
    
    if gmm_male is None or gmm_female is None:
        st.error("Failed to load models. Please check if model files exist.")
        return
    
    # App header
    st.markdown("<h1 class='main-header'>Voice Gender Classification</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Upload an audio file to predict the speaker's gender</p>", unsafe_allow_html=True)
    
    # Information box - using beta_expander instead of expander for older Streamlit versions
    with st.beta_expander("ℹ️ About this app", expanded=False):
        st.markdown("""
        This app uses Gaussian Mixture Models (GMMs) to classify the gender of a speaker from an audio recording.
        
        **Supported file formats:**
        - WAV
        - MP3
        - OGG
        - FLAC
        - M4A
        
        **How it works:**
        1. Upload an audio file containing a voice recording
        2. The app extracts MFCC features from the audio
        3. These features are scored against pre-trained male and female GMM models
        4. The gender with the higher score is predicted
        
        For best results, use clear recordings with minimal background noise.
        """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg", "flac", "m4a"])
    
    # Record audio option
    st.markdown("<p class='sub-header'>Or record audio directly</p>", unsafe_allow_html=True)
    
    # Check if audio_recorder is available in this Streamlit version
    try:
        audio_recording = st.audio_recorder(text="Click to record", pause_threshold=2.0)
    except AttributeError:
        st.warning("Audio recording is not available in your Streamlit version. Please upgrade Streamlit to use this feature.")
        audio_recording = None
    
    # Process the uploaded file or recording
    if uploaded_file is not None:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            audio_path = tmp_file.name
        
        # Display success message
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        
        # Display audio player
        st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
        
        # Display waveform
        waveform = plot_waveform(audio_path)
        if waveform:
            st.pyplot(waveform)
        
        # Add a prediction button
        if st.button("Predict Gender", key="predict_upload"):
            with st.spinner("Analyzing audio..."):
                # Simulate processing time
                time.sleep(1)
                
                # Predict gender
                gender, male_score, female_score = predict_gender(audio_path, gmm_male, gmm_female)
                
                if gender:
                    # Display result
                    st.markdown(f"<h2 class='result-text {'male-result' if gender == 'Male' else 'female-result'}'>Predicted Gender: {gender}</h2>", unsafe_allow_html=True)
                    
                    # Display confidence scores
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Male Score", f"{male_score:.2f}")
                    with col2:
                        st.metric("Female Score", f"{female_score:.2f}")
                    
                    # Display confidence visualization
                    total = abs(male_score) + abs(female_score)
                    male_conf = abs(male_score) / total * 100
                    female_conf = abs(female_score) / total * 100
                    
                    st.markdown("### Confidence Levels")
                    st.progress(male_conf/100, text=f"Male: {male_conf:.1f}%")
                    st.progress(female_conf/100, text=f"Female: {female_conf:.1f}%")
                else:
                    st.error("Failed to predict gender. Please try another audio file.")
        
        # Clean up the temporary file
        os.unlink(audio_path)
    
    elif audio_recording is not None:
        # Create a temporary file for the recording
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_recording)
            audio_path = tmp_file.name
        
        # Display success message
        st.success("Audio recorded successfully!")
        
        # Display audio player
        st.audio(audio_recording, format="audio/wav")
        
        # Display waveform
        waveform = plot_waveform(audio_path)
        if waveform:
            st.pyplot(waveform)
        
        # Add a prediction button
        if st.button("Predict Gender", key="predict_record"):
            with st.spinner("Analyzing audio..."):
                # Simulate processing time
                time.sleep(1)
                
                # Predict gender
                gender, male_score, female_score = predict_gender(audio_path, gmm_male, gmm_female)
                
                if gender:
                    # Display result
                    st.markdown(f"<h2 class='result-text {'male-result' if gender == 'Male' else 'female-result'}'>Predicted Gender: {gender}</h2>", unsafe_allow_html=True)
                    
                    # Display confidence scores
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Male Score", f"{male_score:.2f}")
                    with col2:
                        st.metric("Female Score", f"{female_score:.2f}")
                    
                    # Display confidence visualization
                    total = abs(male_score) + abs(female_score)
                    male_conf = abs(male_score) / total * 100
                    female_conf = abs(female_score) / total * 100
                    
                    st.markdown("### Confidence Levels")
                    st.progress(male_conf/100, text=f"Male: {male_conf:.1f}%")
                    st.progress(female_conf/100, text=f"Female: {female_conf:.1f}%")
                else:
                    st.error("Failed to predict gender. Please try recording again.")
        
        # Clean up the temporary file
        os.unlink(audio_path)
    
    # Footer
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit and scikit-learn")

if __name__ == "__main__":
    main()
