# Import necessary modules
from flask import Flask, request, jsonify
import os
import pickle
import numpy as np
import librosa
import python_speech_features as mfcc
from sklearn import preprocessing
import tempfile
import werkzeug

# Initialize Flask app
app = Flask(__name__)

# Load the GMM models
try:
    gmm_male = pickle.load(open('male.gmm', 'rb'))
    gmm_female = pickle.load(open('female.gmm', 'rb'))
except Exception as e:
    print("Error loading models: {}".format(e))
    # We'll handle this in the API responses

def get_MFCC(sr, audio):
    """
    Extracts the MFCC audio features from a file
    """
    features = mfcc.mfcc(audio, sr, 0.025, 0.01, 13, appendEnergy=False)
    features = preprocessing.scale(features)
    return features

def predict_gender(audio_path, sr=16000):
    """
    Predicts gender from an audio file
    """
    try:
        # Load the audio file
        loading, sr = librosa.core.load(audio_path, sr, float)
        
        # Extract features
        features = get_MFCC(sr, loading)
        
        # Calculate log likelihoods
        log_likelihood_male = np.array(gmm_male.score(features)).sum()
        log_likelihood_female = np.array(gmm_female.score(features)).sum()
        
        # Determine gender based on log likelihood
        if log_likelihood_male >= log_likelihood_female:
            gender = "Male"
        else:
            gender = "Female"
            
        confidence = abs(log_likelihood_male - log_likelihood_female)
        
        return gender, confidence, None
    except Exception as e:
        return None, None, str(e)

@app.route('/')
def home():
    return '''
    <h1>Voice Gender Detection API</h1>
    <p>Use the /predict endpoint to determine the gender from a voice recording.</p>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="audio" accept="audio/*">
        <input type="submit" value="Predict Gender">
    </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    # Check if request has the audio file
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file found'}), 400
    
    file = request.files['audio']
    
    # Check if filename is empty
    if file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400
        
    if file:
        # Create a temporary file
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, werkzeug.utils.secure_filename(file.filename))
        
        # Save the file temporarily
        file.save(temp_path)
        
        # Process the file
        gender, confidence, error = predict_gender(temp_path)
        
        # Clean up the temporary file
        try:
            os.remove(temp_path)
            os.rmdir(temp_dir)
        except:
            pass
        
        if error:
            return jsonify({'error': 'Error processing audio: {}'.format(error)}), 500
        
        # Return the prediction
        return jsonify({
            'gender': gender,
            'confidence': float(confidence),
            'filename': file.filename
        })

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# API endpoint for health check
@app.route('/health', methods=['GET'])
def health_check():
    if 'gmm_male' in globals() and 'gmm_female' in globals():
        return jsonify({'status': 'healthy', 'models_loaded': True})
    else:
        return jsonify({'status': 'unhealthy', 'models_loaded': False}), 503

if __name__ == '__main__':
    # Check if models are loaded
    if 'gmm_male' not in globals() or 'gmm_female' not in globals():
        print("Warning: GMM models not loaded properly. API may not function correctly.")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
