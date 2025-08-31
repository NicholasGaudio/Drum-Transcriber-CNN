import librosa
import numpy as np
import torch

def audio_to_melspectrogram(audio_path):
    duration = .5
    amplitudes, sample_rate = librosa.load(audio_path, sr = 22050, duration = duration) # load audio, create 1d array of amplitudes

    audio_length = len(amplitudes) / sample_rate
    if audio_length < duration:
        padding = np.zeros(int((duration - audio_length) * sample_rate))
        amplitudes = np.concatenate((amplitudes, padding)) # pad with zeros to make it duration time long

    mel  = librosa.feature.melspectrogram(y=amplitudes, sr=sample_rate, n_mels=128, fmax=8000) # create mel spectrogram
    mel_decibals = librosa.power_to_db(mel , ref=np.max) # convert to decibels
    mel_decibals = torch.tensor(mel, dtype=torch.float32)
    mel_decibals = mel_decibals.unsqueeze(0)

    return mel_decibals
