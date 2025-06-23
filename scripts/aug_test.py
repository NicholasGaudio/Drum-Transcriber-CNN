import librosa
from data_augmentation import add_white_noise, pitch_shift, amplitude_scaling
import soundfile as sf

def main():
    audio_path = r"..\data\snare\snare.mp3"  
    amplitudes, sample_rate = librosa.load(audio_path, sr=22050, duration=0.5)

    try:
        white_noise = add_white_noise(amplitudes)
        sf.write('augmented_white_noise.wav', white_noise, sample_rate)
    except Exception as e:
        print(f"Error adding white noise: {e}")

if __name__ == "__main__":
    main()
    
