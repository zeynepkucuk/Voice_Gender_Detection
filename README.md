# Voice Gender Detection

A machine learning application that detects gender from voice recordings using Gaussian Mixture Models (GMM). This project includes both a command-line interface and a REST API for gender prediction from audio files.

![Voice Gender Detection](https://img.shields.io/badge/ML-Voice%20Gender%20Detection-blue)
![Python](https://img.shields.io/badge/Python-3.x-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Command Line Interface](#command-line-interface)
  - [Web API](#web-api)
- [Technical Details](#technical-details)
  - [Feature Extraction](#feature-extraction)
  - [Machine Learning Model](#machine-learning-model)
- [API Documentation](#api-documentation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

This application uses audio processing techniques and machine learning to determine whether a voice recording is of a male or female speaker. It extracts Mel-Frequency Cepstral Coefficients (MFCC) features from audio samples and uses pre-trained Gaussian Mixture Models to classify the gender of the speaker.

## Features

- **Gender prediction** from voice recordings with confidence scores
- **Command-line interface** for quick predictions
- **REST API** for integration into other applications
- **Web interface** for easy testing via browser
- **Support for various audio formats** (WAV, MP3, etc.)

## Requirements

- Python 3.x (recommended) or Python 2.7
- Required Python packages (listed in `requirements.txt`):
  - numpy, matplotlib - For data manipulation and visualization
  - scipy, librosa, python_speech_features - For audio processing and feature extraction
  - scikit-learn - For machine learning models and preprocessing
  - flask, werkzeug - For the web API
  - sounddevice, soundfile - For audio recording and processing
- Pre-trained GMM models (`male.gmm` and `female.gmm`)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Voice_Gender_Detection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   If you're using Python 2:
   ```bash
   pip2 install -r requirements.txt
   ```

   Or with Python 3:
   ```bash
   pip3 install -r requirements.txt
   ```

3. Ensure the GMM model files (`male.gmm` and `female.gmm`) are in the project directory.

## Usage

### Command Line Interface

To analyze a voice recording using the command line:

```bash
python Gender.py /path/to/your/audiofile.wav
```

Example:

```bash
python Gender.py female.wav
```

The output will show the predicted gender:

```
--------------------------------------------RESULT---------------------------------- 

female.wav Gender: Female
```

### Web API

1. Start the API server:
   ```bash
   python app.py
   ```

2. The server will run on `http://localhost:5000` by default.

3. Access the web interface by opening a browser and navigating to `http://localhost:5000`.

4. To make API calls programmatically:

   Using cURL:
   ```bash
   curl -X POST -F "audio=@/path/to/your/audiofile.wav" http://localhost:5000/predict
   ```

   Using Python:
   ```python
   import requests

   url = "http://localhost:5000/predict"
   files = {"audio": open("voice_sample.wav", "rb")}
   response = requests.post(url, files=files)
   print(response.json())
   ```

## Technical Details

### Feature Extraction

The application uses Mel-Frequency Cepstral Coefficients (MFCC) for audio feature extraction:

1. Audio is processed in short frames (25ms with 10ms shift)
2. 13 MFCC features are extracted from each frame
3. Features are scaled using scikit-learn's preprocessing

```python
def get_MFCC(sr, audio):
    features = mfcc.mfcc(audio, sr, 0.025, 0.01, 13, appendEnergy=False)
    features = preprocessing.scale(features)
    return features
```

### Machine Learning Model

Gender detection uses pre-trained Gaussian Mixture Models:

1. Two GMM models are used: one for male voices and one for female voices
2. For classification, log-likelihood scores are computed for each model
3. The gender with the higher log-likelihood score is selected as the prediction
4. The absolute difference between scores provides a confidence measure

## API Documentation

The project includes a Flask-based REST API for voice gender detection.

### API Endpoints

#### 1. Home Page

- **URL**: `/`
- **Method**: `GET`
- **Description**: Returns a simple HTML page with instructions and a form to test the API.

#### 2. Predict Gender

- **URL**: `/predict`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `audio`: Audio file to analyze (WAV, MP3, etc.)
- **Response Format**: JSON
- **Successful Response Example**:
  ```json
  {
    "gender": "Male",
    "confidence": 42.57,
    "filename": "voice_sample.wav"
  }
  ```

## Project Structure

```
Voice_Gender_Detection/
├── Gender.py            # Command-line script for gender detection
├── app.py               # Flask API implementation
├── requirements.txt     # Python dependencies
├── README.md            # This documentation file
├── male.gmm             # Pre-trained model for male voice
└── female.gmm           # Pre-trained model for female voice
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named flask**
   
   Solution: Install Flask package
   ```bash
   pip install flask
   ```

2. **Models not loading**
   
   Solution: Ensure `male.gmm` and `female.gmm` files are in the same directory as the Python scripts.

3. **Audio processing errors**
   
   Solution: Make sure audio files are in supported formats and not corrupted. Try using WAV files for best compatibility.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
