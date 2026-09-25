from pathlib import Path
import csv

dataset = Path(r"C:\DSP\ESC-50-master")
audio_folder = dataset / "audio"
csv_file = dataset / "meta" / "esc50.csv"

# Map ESC-50 categories to our project classes
category_map = {
    "door_wood_knock": "door_knocking",
    "glass_breaking": "glass_breaking",
    "clock_alarm": "alarm_siren",
    "siren": "alarm_siren",
    "dog": "dog_barking",
    "footsteps": "footsteps",
    "rain": "background_noise",
    "wind": "background_noise",
    "sea_waves": "background_noise",
}

with open(csv_file, newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

output_rows = []

for row in rows:
    category = row["category"]

    if category in category_map:
        output_rows.append({
            "filename": row["filename"],
            "category": category_map[category],
            "fold": row["fold"],
            "filepath": str(audio_folder / row["filename"])
        })

output_file = dataset / "training_data.csv"

with open(output_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["filename", "category", "fold", "filepath"]
    )
    writer.writeheader()
    writer.writerows(output_rows)

print("Selected recordings:", len(output_rows))
print("Saved to:", output_file)