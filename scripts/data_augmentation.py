import librosa
import numpy as np

def add_white_noise(amplitudes, noise_factor=0.01):
    noise = np.random.randn(len(amplitudes))
    augmented_data = amplitudes + noise_factor * noise
    return augmented_data

def pitch_shift(amplitudes, sample_rate, n_steps):
    return librosa.effects.pitch_shift(amplitudes, sample_rate, n_steps)

def amplitude_scaling(amplitudes, scale):
    return amplitudes * scale