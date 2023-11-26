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



gmm_male = pickle.load(open('male.gmm', 'rb'))
gmm_female = pickle.load(open('female.gmm', 'rb'))



def get_MFCC(sr,audio):

    features = mfcc.mfcc(audio, sr, 0.025, 0.01, 13, appendEnergy = False)
    features = preprocessing.scale(features)
    
    return features



def get_features(source):
    sr = 16000 


    # Split files
    files = [os.path.join(source,f) for f in os.listdir(source) if f.endswith('.wav')]
    test_files = files
    print(len(test_files))

    # Test features  
    features_test = []
    for filename in test_files:
        print(filename)

        #sr, audio = read(f)
        audio, sr = librosa.core.load(filename, sr, float)
        vector = get_MFCC(sr,audio)


        """
        if len(features_test) == 0:
            features_test = vector
        else:
            features_test = np.vstack((features_test, vector))

        """
        return  vector
        return filename



#source = "AudioSet/male_clips"
#features_test_male = get_features(source)


#source = "AudioSet/female_clips"
source = "/Users/app/Documents/GitHub/voiceGenderDetection/yuklendi20December"
features_test_female =  get_features(source)



"""
output = []

for f in features_test_male:

    log_likelihood_male = np.array(gmm_male.score([f])).sum()
    log_likelihood_female = np.array(gmm_female.score([f])).sum()
    
    if log_likelihood_male > log_likelihood_female:
        output.append(0)
    else:
        output.append(1)
"""

"""
accuracy_male = (1 - sum(output)/len(output))

print(sum(output),len(output))

print("Male Clips Accuracy: ", accuracy_male)
"""


output = []

for f in features_test_female:
    
    log_likelihood_male = np.array(gmm_male.score([f])).sum()
    log_likelihood_female = np.array(gmm_female.score([f])).sum()
    
    if log_likelihood_male > log_likelihood_female:
        print("Male")
        output.append(0)
    else:
        print("Female")
        output.append(1)



"""
accuracy_female = (sum(output)/len(output))

print(sum(output),len(output))

print("Female Clips Accuracy: ", accuracy_female)
"""

