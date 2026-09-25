from pathlib import Path
import csv
import numpy as np
import librosa
import joblib

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Dataset locations
dataset = Path(r"C:\DSP\ESC-50-master")
csv_file = dataset / "training_data.csv"

# Extract DSP features from one audio file
def extract_features(filepath):
    y, sr = librosa.load(filepath, sr=22050)

    # MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Additional audio features
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        [np.mean(centroid), np.std(centroid)],
        [np.mean(zcr), np.std(zcr)],
        [np.mean(rolloff), np.std(rolloff)]
    ])

    return features


# Read selected recordings
with open(csv_file, newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

X_train = []
y_train = []
X_test = []
y_test = []

print("Extracting audio features...")

for row in rows:
    filepath = row["filepath"]
    label = row["category"]
    fold = int(row["fold"])

    features = extract_features(filepath)

    # Use fold 5 for testing; folds 1-4 for training
    if fold == 5:
        X_test.append(features)
        y_test.append(label)
    else:
        X_train.append(features)
        y_train.append(label)

print("Training recordings:", len(X_train))
print("Testing recordings:", len(X_test))

# Create and train the classifier
model = make_pipeline(
    StandardScaler(),
    SVC(
    kernel="rbf",
    class_weight="balanced",
    probability=True
    )
)

print("Training model...")
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, predictions))
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

# Save trained model
joblib.dump(model, "sound_classifier.pkl")
print("\nModel saved as sound_classifier.pkl")