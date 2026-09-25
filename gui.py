
import threading
import queue
import tkinter as tk
from tkinter import ttk

import numpy as np
import librosa
import joblib
import sounddevice as sd
import winsound


SAMPLE_RATE = 22050
DURATION = 5
ALERT_THRESHOLD = 65

alert_classes = {"glass_breaking", "alarm_siren"}

model = joblib.load("sound_classifier.pkl")

result_queue = queue.Queue()
stop_event = threading.Event()
listener_thread = None


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


def listen_loop():
    while not stop_event.is_set():
        try:
            recording = sd.rec(
                int(DURATION * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32"
            )
            sd.wait()

            if stop_event.is_set():
                break

            audio = recording.flatten()
            features = extract_features(audio, SAMPLE_RATE).reshape(1, -1)

            probabilities = model.predict_proba(features)[0]
            classes = model.classes_
            top_indices = np.argsort(probabilities)[::-1][:3]

            prediction = classes[top_indices[0]]
            confidence = probabilities[top_indices[0]] * 100

            top_results = [
                f"{classes[i]}: {probabilities[i] * 100:.1f}%"
                for i in top_indices
            ]

            is_alert = (
                prediction in alert_classes
                and confidence >= ALERT_THRESHOLD
            )

            result_queue.put(
                (prediction, confidence, top_results, is_alert)
            )

            if is_alert:
                winsound.Beep(1200, 700)

        except Exception as error:
            result_queue.put(("ERROR", str(error), [], False))
            break


def start_listening():
    global listener_thread

    if listener_thread and listener_thread.is_alive():
        return

    stop_event.clear()
    listener_thread = threading.Thread(
        target=listen_loop,
        daemon=True
    )
    listener_thread.start()

    status_label.config(text="Listening...", foreground="green")
    start_button.config(state="disabled")
    stop_button.config(state="normal")


def stop_listening():
    stop_event.set()

    status_label.config(text="Stopping...", foreground="orange")
    start_button.config(state="normal")
    stop_button.config(state="disabled")


def update_display():
    while not result_queue.empty():
        prediction, confidence, top_results, is_alert = result_queue.get()

        if prediction == "ERROR":
            prediction_label.config(text="Error")
            confidence_label.config(text=str(confidence))
            status_label.config(text="Check terminal for details")
            continue

        prediction_label.config(text=prediction.replace("_", " ").title())
        confidence_label.config(text=f"{confidence:.1f}%")

        top_text = "\n".join(top_results)
        top_label.config(text=top_text)

        if is_alert:
            alert_label.config(
                text="⚠ ALERT: Suspicious sound detected!",
                foreground="red"
            )
        else:
            alert_label.config(
                text="No alert",
                foreground="gray"
            )

    if stop_event.is_set():
        if listener_thread is None or not listener_thread.is_alive():
            status_label.config(text="Stopped", foreground="gray")

    root.after(200, update_display)


def close_app():
    stop_event.set()
    root.destroy()


root = tk.Tk()
root.title("Smart Audio Surveillance System")
root.geometry("520x430")
root.resizable(False, False)

title_label = ttk.Label(
    root,
    text="Smart Audio Surveillance System",
    font=("Arial", 17, "bold")
)
title_label.pack(pady=15)

status_label = ttk.Label(
    root,
    text="Ready to listen",
    font=("Arial", 11)
)
status_label.pack(pady=5)

prediction_label = ttk.Label(
    root,
    text="Waiting for sound...",
    font=("Arial", 20, "bold")
)
prediction_label.pack(pady=10)

confidence_label = ttk.Label(
    root,
    text="Confidence: --",
    font=("Arial", 12)
)
confidence_label.pack()

buttons_frame = ttk.Frame(root)
buttons_frame.pack(pady=15)

start_button = ttk.Button(
    buttons_frame,
    text="Start Listening",
    command=start_listening
)
start_button.grid(row=0, column=0, padx=10)

stop_button = ttk.Button(
    buttons_frame,
    text="Stop Listening",
    command=stop_listening,
    state="disabled"
)
stop_button.grid(row=0, column=1, padx=10)

alert_label = ttk.Label(
    root,
    text="No alert",
    font=("Arial", 13, "bold")
)
alert_label.pack(pady=5)

ttk.Label(
    root,
    text="Top predictions:",
    font=("Arial", 11, "bold")
).pack(pady=(8, 2))

top_label = ttk.Label(
    root,
    text="--",
    justify="center"
)
top_label.pack()

root.protocol("WM_DELETE_WINDOW", close_app)
root.after(200, update_display)
root.mainloop()
