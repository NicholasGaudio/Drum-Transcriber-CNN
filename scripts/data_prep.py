import librosa
import numpy as np
import os

def audio_to_melspectrogram(audio_path):
    amplitudes, sample_rate = librosa.load(audio_path, sr = 22050) # load audio, create 1d array of amplitudes
    melspectrogram  = librosa.feature.melspectrogram(y=amplitudes, sr=sample_rate, n_mels=128, fmax=8000) # create mel spectrogram
    melspectrogram_decibals = librosa.power_to_db(melspectrogram , ref=np.max) # convert to decibels
    return melspectrogram_decibals
