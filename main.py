
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load the knock sound
audio, sr = librosa.load("audio/knock.wav", sr=None)

# Display audio information
print("Sampling frequency:", sr, "Hz")
print("Number of samples:", len(audio))
print("Duration:", len(audio) / sr, "seconds")

# Plot the waveform
plt.figure(figsize=(10, 4))
librosa.display.waveshow(audio, sr=sr)

plt.title("Knock Sound Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()

plt.show()


# Generate the spectrogram
import numpy as np

D = librosa.stft(audio)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

plt.figure(figsize=(10, 4))

librosa.display.specshow(
    S_db,
    sr=sr,
    x_axis="time",
    y_axis="hz"
)

plt.colorbar(format="%+2.0f dB")
plt.title("Knock Sound Spectrogram")
plt.xlabel("Time (seconds)")
plt.ylabel("Frequency (Hz)")

plt.show()