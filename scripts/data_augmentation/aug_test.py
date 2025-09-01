import librosa
from scripts.data_augmentation.augmentation_functions import add_white_noise, custom_pitch_shift, amplitude_scaling
import soundfile as sf
import os
from scripts.env import directory
import numpy as np

data_dir = directory + "\snare"

for root, dirs, files in os.walk(data_dir):
    for file in files:
        if "AUGMENTED" in file:
            continue
        else:
            file_path = os.path.join(root, file)
            file_name = os.path.splitext(file)[0]

            amplitudes, sample_rate = librosa.load(file_path, sr=22050)
            for i in np.arange(0.001, 0.005, 0.001):
                augmented = add_white_noise(amplitudes, noise_factor=i)
                output_path = os.path.join(root, f"{file_name}_WHITE_NOISE_{i: .3f}_AUGMENTED.wav")
                sf.write(output_path, augmented, sample_rate)
            for i in range(-8, 9):
                if i == 0:
                    continue
                augmented = custom_pitch_shift(amplitudes, sample_rate, n_steps=i)
                output_path = os.path.join(root, f"{file_name}_PITCH_SHIFT_{i}_AUGMENTED.wav")
                sf.write(output_path, augmented, sample_rate)
            for i in np.arange(.5, 1.51, .1):
                augmented = amplitude_scaling(amplitudes, scale=i)
                output_path = os.path.join(root, f"{file_name}_AMPLITUDE_SCALING_{i:.1f}_AUGMENTED.wav")
                sf.write(output_path, augmented, sample_rate)
    
