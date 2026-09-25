import time
import numpy as np
import librosa
import joblib
import sounddevice as sd
import winsound

SAMPLE_RATE = 22050
DURATION = 5  # seconds

# Load the trained classifier
model = joblib.load("sound_classifier.pkl")


# Extract the same DSP features used during training
def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    return np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        [np.mean(centroid), np.std(centroid)],
        [np.mean(zcr), np.std(zcr)],
        [np.mean(rolloff), np.std(rolloff)]
    ])


alert_classes = {"glass_breaking", "alarm_siren"}

print("LIVE SOUND SURVEILLANCE STARTED")
print("Listening through your microphone...")
print("Press Ctrl + C to stop.\n")

try:
    while True:
        print("Listening for 5 seconds...")

        # Record audio from the default microphone
        recording = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )
        sd.wait()

        # Convert stereo-shaped array to a 1D signal
        audio = recording.flatten()

        # Extract features and predict
        features = extract_features(audio, SAMPLE_RATE).reshape(1, -1)

        probabilities = model.predict_proba(features)[0]
        classes = model.classes_
        top_indices = np.argsort(probabilities)[::-1][:3]

        print("\nTop predictions:")
        for i in top_indices:
            print(f"{classes[i]}: {probabilities[i] * 100:.1f}%")

        prediction = classes[top_indices[0]]
        confidence = probabilities[top_indices[0]] * 100

        print(f"Detected: {prediction}")
        print(f"Estimated confidence: {confidence:.1f}%")

        # Alert only for selected sounds with confidence above 65%
        if prediction in alert_classes and confidence >= 65:
            print("⚠️ ALERT: Suspicious sound detected!")
            winsound.Beep(1200, 700)
        else:
            print("No alert.")

        print("-" * 35)

except KeyboardInterrupt:
    print("\nSurveillance stopped.")