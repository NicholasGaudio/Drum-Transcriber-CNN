import librosa
import numpy as np

def audio_to_melspectrogram(audio_path):
    duration = .5
    amplitudes, sample_rate = librosa.load(audio_path, sr = 22050, duration = duration) # load audio, create 1d array of amplitudes

    audio_length = len(amplitudes) / sample_rate
    if audio_length < duration:
        padding = np.zeros(int((duration - audio_length) * sample_rate))
        amplitudes = np.concatenate((amplitudes, padding)) # pad with zeros to make it duration time long

    melspectrogram  = librosa.feature.melspectrogram(y=amplitudes, sr=sample_rate, n_mels=128, fmax=8000) # create mel spectrogram
    melspectrogram_decibals = librosa.power_to_db(melspectrogram , ref=np.max) # convert to decibels
    melspectrogram_decibals = np.expand_dims(melspectrogram_decibals, axis=-1) # add channel dimension for CNN, now the shape is (128, 128, 1)

    return melspectrogram_decibals
