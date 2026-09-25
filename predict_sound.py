import librosa
import numpy as np
import joblib
import winsound

def extract_features(filepath):
    y, sr = librosa.load(filepath, sr=22050)

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

model = joblib.load("sound_classifier.pkl")

audio_file = input("Enter the WAV file path: ")

features = extract_features(audio_file).reshape(1, -1)

# Predict class and estimated probability
prediction = model.predict(features)[0]
probabilities = model.predict_proba(features)[0]
confidence = np.max(probabilities) * 100

print("\nDetected sound:", prediction)
print(f"Estimated confidence: {confidence:.1f}%")

# Sounds that trigger an alert
alert_classes = {
    "glass_breaking",
    "alarm_siren"
}

if prediction in alert_classes:
    print("ALERT! Suspicious sound detected!")
    winsound.Beep(1200, 700)
else:
    print("No alert triggered.")