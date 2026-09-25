import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 22050
DURATION = 5

print("Recording for 5 seconds...")
recording = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"
)
sd.wait()

sf.write("mic_test.wav", recording, SAMPLE_RATE)

print("Saved recording as mic_test.wav")