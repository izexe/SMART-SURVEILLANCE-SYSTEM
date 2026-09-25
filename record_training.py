from pathlib import Path
from datetime import datetime
import csv
import sounddevice as sd
import soundfile as sf

# Project folders
PROJECT = Path(r"C:\DSP")
RECORDINGS_FOLDER = PROJECT / "mic_data"
CSV_FILE = PROJECT / "ESC-50-master" / "training_data.csv"

SAMPLE_RATE = 22050
DURATION = 5  # seconds

# These labels must match the classifier's existing labels
CLASSES = [
    "door_knocking",
    "glass_breaking",
    "alarm_siren",
    "dog_barking",
    "footsteps",
    "background_noise"
]

RECORDINGS_FOLDER.mkdir(exist_ok=True)

print("MICROPHONE TRAINING RECORDER")
print("Available classes:")
for label in CLASSES:
    print("-", label)

print("\nEach recording will be 5 seconds long.")
print("Play the sound near your microphone while recording.")
print("Type 'q' to quit.\n")

while True:
    label = input("Enter sound class: ").strip().lower()

    if label == "q":
        print("Recorder closed.")
        break

    if label not in CLASSES:
        print("Invalid class. Please choose one from the list.")
        continue

    input(f"\nGet ready to record {label}. Press Enter to start...")

    print("Recording now!")
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()

    # Create a folder for this sound class
    class_folder = RECORDINGS_FOLDER / label
    class_folder.mkdir(parents=True, exist_ok=True)

    # Unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{label}_{timestamp}.wav"
    filepath = class_folder / filename

    # Save the WAV recording
    sf.write(filepath, audio, SAMPLE_RATE)

    # Add it to the existing training CSV
    file_exists = CSV_FILE.exists()

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["filename", "category", "fold", "filepath"]
        )

        if not file_exists or CSV_FILE.stat().st_size == 0:
            writer.writeheader()

        writer.writerow({
            "filename": filename,
            "category": label,
            "fold": 1,
            "filepath": str(filepath)
        })

    print(f"Saved: {filepath}")
    print("Added to training_data.csv\n")