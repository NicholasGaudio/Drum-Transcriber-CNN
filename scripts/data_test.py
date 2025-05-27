from data_prep import audio_to_melspectrogram
import matplotlib.pyplot as plt

melspectrogram = audio_to_melspectrogram("data/kick.wav")

plt.imshow(melspectrogram, aspect='auto', origin='lower')
plt.colorbar(format='%+2.0f dB')
plt.title("Mel Spectrogram")
plt.xlabel("Time")
plt.ylabel("Mel bands")
plt.show()
