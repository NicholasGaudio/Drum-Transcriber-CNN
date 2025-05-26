from data_prep import audio_to_melspectrogram
import matplotlib.pyplot as plt

spectrogram = audio_to_melspectrogram("data/closed-hat.wav")

plt.imshow(spectrogram, aspect='auto', origin='lower')
plt.colorbar(format='%+2.0f dB')
plt.title("Mel Spectrogram")
plt.xlabel("Time")
plt.ylabel("Mel bands")
plt.show()
