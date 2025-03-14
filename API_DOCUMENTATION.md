# Voice Gender Detection API Documentation

This API provides gender detection from voice recordings using Gaussian Mixture Model (GMM) classifiers.

## Installation and Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Ensure the GMM model files (`male.gmm` and `female.gmm`) are in the same directory as the app.py file.

3. Run the API server:
   ```
   python app.py
   ```

4. By default, the server runs on `http://localhost:5000`

## API Endpoints

### 1. Home Page

- **URL**: `/`
- **Method**: `GET`
- **Description**: Returns a simple HTML page with instructions and a form to test the API.

### 2. Predict Gender

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
- **Error Response Example**:
  ```json
  {
    "error": "No audio file found"
  }
  ```

### 3. Health Check

- **URL**: `/health`
- **Method**: `GET`
- **Description**: Checks if the API and models are loaded properly
- **Response Example**:
  ```json
  {
    "status": "healthy",
    "models_loaded": true
  }
  ```

## Usage Examples

### Using cURL

```bash
curl -X POST -F "audio=@/path/to/your/audio.wav" http://localhost:5000/predict
```

### Using Python Requests

```python
import requests

url = "http://localhost:5000/predict"
files = {"audio": open("voice_sample.wav", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

### Using HTML Form

You can also use the form on the homepage at `http://localhost:5000/` to upload and test your audio files directly from a web browser.

## Error Handling

The API returns appropriate HTTP status codes:

- `200 OK`: Request was successful
- `400 Bad Request`: Missing or invalid audio file
- `500 Internal Server Error`: Server-side processing error
- `503 Service Unavailable`: Models not loaded properly

## Notes

- Supported audio formats depend on the librosa library (WAV, MP3, FLAC, etc.)
- For best results, use clear audio recordings with minimal background noise
- The confidence value represents the absolute difference between male and female log-likelihoods