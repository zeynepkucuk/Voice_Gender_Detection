
# Data manipulation
import numpy as np
import matplotlib.pyplot as plt

# Feature extraction
import scipy
import librosa
import python_speech_features as mfcc
import os
from scipy.io.wavfile import read

# Model training
from sklearn.mixture import GaussianMixture as GMM
from sklearn import preprocessing
import pickle

# Live recording
import sounddevice as sd
import soundfile as sf
#argument
import sys


"python3 Gender.py female.wav"


def get_MFCC(sr,audio):
    """
    Extracts the MFCC audio features from a file
    """
    features = mfcc.mfcc(audio, sr, 0.025, 0.01, 13, appendEnergy = False)
    features = preprocessing.scale(features)
    return features


def load_and_predict(gmm_male, gmm_female, sr=16000, channels=1, filename= sys.argv[1]):


    loading, sr = librosa.core.load(filename, sr, float)
    features =get_MFCC(sr,loading)

    
    features = get_MFCC(sr,loading)
    scores = None


    log_likelihood_male = np.array(gmm_male.score(features)).sum()
    log_likelihood_female = np.array(gmm_female.score(features)).sum()

    if log_likelihood_male >= log_likelihood_female:
        return("Male")
    else:
        return("Female")



gmm_male = pickle.load(open('male.gmm', 'rb'))
gmm_female = pickle.load(open('female.gmm', 'rb'))

gender = load_and_predict(gmm_male, gmm_female)

print("--------------------------------------------RESULT---------------------------------- \n")


print(sys.argv[1] + " Gender : " + gender )





