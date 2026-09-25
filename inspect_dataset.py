from pathlib import Path
import csv
from collections import Counter

dataset = Path(r"C:\DSP\ESC-50-master")
audio_folder = dataset / "audio"
csv_file = dataset / "meta" / "esc50.csv"

with open(csv_file, newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

counts = Counter(row["category"] for row in rows)

print("Total labeled recordings:", len(rows))
print("Audio files found:", len(list(audio_folder.glob("*.wav"))))

print("\nSound categories:")
for category, count in sorted(counts.items()):
    print(category, ":", count)

missing = [
    row["filename"]
    for row in rows
    if not (audio_folder / row["filename"]).exists()
]

print("\nMissing audio files:", len(missing))