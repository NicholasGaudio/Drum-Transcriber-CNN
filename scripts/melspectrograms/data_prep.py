import librosa
import numpy as np
import torch

def audio_to_melspectrogram(audio_path, duration=0.5, n_mels=128, target_width=128):
    # Load audio, fixed duration
    amplitudes, sample_rate = librosa.load(audio_path, sr=22050, duration=duration)

    # Pad if shorter than desired duration
    audio_length = len(amplitudes) / sample_rate
    if audio_length < duration:
        padding = np.zeros(int((duration - audio_length) * sample_rate))
        amplitudes = np.concatenate((amplitudes, padding))

    # Create mel spectrogram
    mel = librosa.feature.melspectrogram(y=amplitudes, sr=sample_rate, n_mels=n_mels, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Fix time axis (width) to target_width
    if mel_db.shape[1] < target_width:
        # pad with zeros
        mel_db = np.pad(mel_db, ((0,0), (0, target_width - mel_db.shape[1])), mode='constant')
    else:
        # truncate
        mel_db = mel_db[:, :target_width]

    # Add channel dimension for CNN
    mel_db = mel_db[np.newaxis, :, :]  # shape: (1, n_mels, target_width)
    mel_tensor = torch.tensor(mel_db, dtype=torch.float32)

    return mel_tensor
