# Smart Audio Surveillance System

A Python-based audio classification project that listens to microphone
input, identifies selected environmental sounds, and raises an alert
when it detects a possible glass-breaking or alarm-siren sound.

This is an educational DSP and machine-learning prototype built to
explore how digital audio signals can be sampled, processed, and
classified using a laptop microphone.

## Features

-   Classifies audio into six sound categories:
    -   Door knocking
    -   Glass breaking
    -   Alarm / siren
    -   Dog barking
    -   Footsteps
    -   Background nature sounds (rain, wind, and sea waves)
-   Extracts digital signal processing (DSP) features from audio.
-   Trains an SVM classifier using the ESC-50 dataset and additional
    microphone recordings.
-   Supports prediction from a WAV file.
-   Supports live microphone detection in five-second audio blocks.
-   Includes a simple Tkinter graphical user interface (GUI).
-   Plays an alert beep for glass-breaking or alarm/siren predictions
    when the confidence threshold is met.

## Project Workflow

``` text
Microphone / WAV audio
        ↓
Audio sampling (mono, 22,050 Hz)
        ↓
Preprocessing and feature extraction
        ↓
32-dimensional feature vector
        ↓
Trained SVM classifier
        ↓
Predicted sound class + confidence
        ↓
GUI display / alert beep
```

## Sound Classes and Dataset

The project uses selected categories from the [ESC-50
dataset](https://github.com/karolpiczak/ESC-50). The selected classes
are:

  Project label        ESC-50 category/categories
  -------------------- -----------------------------
  `door_knocking`      `door_wood_knock`
  `glass_breaking`     `glass_breaking`
  `alarm_siren`        `clock_alarm`, `siren`
  `dog_barking`        `dog`
  `footsteps`          `footsteps`
  `background_noise`   `rain`, `wind`, `sea_waves`

The dataset contains labelled environmental audio recordings. Additional
recordings captured using a microphone can be added to improve how the
model handles sounds recorded in the project's actual environment.

**Note:** The `background_noise` class represents selected nature
sounds, not every type of room or environmental noise.

## DSP Concepts Used

### 1. Sampling

The audio is converted to mono and sampled at **22,050 samples per
second (22.05 kHz)**.

For a five-second audio block:

-   Sampling frequency: 22,050 Hz
-   Number of samples: 22,050 × 5 = **110,250 samples**
-   Nyquist frequency: 22,050 / 2 = **11,025 Hz**

The Nyquist frequency is the highest frequency that can theoretically be
represented without aliasing under ideal sampling conditions.

### 2. Feature Extraction

The model does not classify the raw audio waveform directly. It extracts
32 numerical features from each recording:

  ------------------------------------------------------------------------
  Feature                         Number of features What it represents
  --------------------- ---------------------------- ---------------------
  MFCC mean                                       13 Average cepstral
                                                     characteristics of
                                                     the sound

  MFCC standard                                   13 Variation of cepstral
  deviation                                          characteristics over
                                                     time

  Spectral centroid                                2 Brightness / spectral
  mean and standard                                  centre of gravity
  deviation                                          

  Zero-crossing rate                               2 How frequently the
  mean and standard                                  waveform changes sign
  deviation                                          

  Spectral rolloff mean                            2 Frequency below which
  and standard                                       a selected portion of
  deviation                                          spectral energy lies

  **Total**                                   **32** Feature vector
                                                     supplied to the
                                                     classifier
  ------------------------------------------------------------------------

MFCCs (Mel-Frequency Cepstral Coefficients) are calculated from
short-time audio frames using spectral analysis, a Mel filter bank,
logarithmic energy, and a discrete cosine transform (DCT).

The mean and standard deviation summarize how these features behave
across the audio block.

### 3. Classification

The extracted feature vector is passed to a machine-learning pipeline
consisting of:

1.  `StandardScaler` to scale the features.
2.  An SVM classifier with an RBF kernel.

The classifier predicts one of the six known sound categories. The
displayed confidence is based on the model's `predict_proba` output; it
should be treated as an estimate, not a guarantee.

### 4. Alert

The live detector and GUI can raise an alert for:

-   `glass_breaking`
-   `alarm_siren`

The alert is triggered when the predicted class is one of these
categories and the estimated confidence is at least **65%**. The beep is
generated using Python's built-in `winsound` module on Windows.

## Requirements

-   Windows
-   Python 3
-   VS Code or another Python editor
-   Laptop microphone
-   ESC-50 dataset for training

Python packages used:

``` bash
pip install numpy scipy matplotlib librosa soundfile scikit-learn joblib sounddevice
```

`tkinter` and `winsound` are included with standard Windows Python
installations in typical setups. `winsound` is Windows-specific.

## Setup

### 1. Clone the repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

Replace the placeholders with your repository URL and folder name.

### 2. Install dependencies

``` bash
py -m pip install numpy scipy matplotlib librosa soundfile scikit-learn joblib sounddevice
```

### 3. Download the ESC-50 dataset

Download or clone the ESC-50 dataset from:

https://github.com/karolpiczak/ESC-50

Extract it so the project can access:

``` text
ESC-50-master/
├── audio/
└── meta/
    └── esc50.csv
```

The data preparation and training scripts expect the dataset location to
match the paths configured in the scripts. Update those paths if your
folders are arranged differently.

### 4. Prepare the selected data

Run:

``` bash
py prepare_data.py
```

This creates `training_data.csv` containing the selected sound
categories.

### 5. Train the model

Run:

``` bash
py train_model.py
```

The script extracts features, trains the classifier, evaluates it using
the configured ESC-50 fold split, and saves the trained model as:

``` text
sound_classifier.pkl
```

### 6. Run a WAV-file prediction

``` bash
py predict_sound.py
```

Follow the script's prompt or configured file path to choose a WAV
recording.

### 7. Run live detection

``` bash
py live_detect.py
```

Allow microphone access if Windows asks for permission. The program
processes audio in five-second blocks. Press **Ctrl+C** in the terminal
to stop it.

### 8. Run the GUI

``` bash
py gui.py
```

Use the Start and Stop controls to begin or end microphone monitoring.
The GUI displays the predicted sound and the top prediction results.

## Adding Microphone Recordings

The project can use additional microphone recordings stored by class.
For useful training examples:

-   Record different examples of each class.
-   Vary distance, room, and background conditions.
-   Listen to each recording and make sure its label is correct.
-   Keep some recordings separate for testing; do not train on the same
    recordings used to evaluate microphone performance.

The project's training CSV can include the added recordings, but the
paths and labels must match the format expected by the scripts.

## Model Evaluation and Limitations

The initial ESC-50 held-out-fold evaluation was approximately **90.3%
accuracy on 72 test clips**. This result is specific to that dataset
split and should not be interpreted as expected real-world microphone
accuracy.

Important limitations:

-   ESC-50 recordings and laptop microphone recordings can sound
    different, so performance may change in real environments.
-   The classifier always chooses among the six known categories; it
    does not currently have a fully validated "unknown sound" class.
-   The probability value is an estimate and is not a calibrated
    guarantee.
-   Audio is processed in five-second blocks, so detection is not
    instantaneous and may have several seconds of delay.
-   The nature-sound background class does not cover all possible
    background noise.
-   This is a student prototype for learning and demonstration. It is
    **not a certified or dependable emergency/security alarm** and
    should not be relied on for personal safety or property protection.

## Suggested Repository Structure

Your exact filenames may vary. A typical layout is:

``` text
Smart-Audio-Surveillance-System/
├── prepare_data.py
├── train_model.py
├── predict_sound.py
├── live_detect.py
├── gui.py
├── record_training.py
├── requirements.txt
├── README.md
└── .gitignore
```

The ESC-50 dataset, generated CSV files, model file, and personal
microphone recordings may be large or contain data you do not want to
publish. Consider keeping them out of Git and documenting how to obtain
or generate them instead.

## Future Improvements

-   Evaluate microphone recordings using a separate held-out test set.
-   Add a confusion matrix and per-class precision, recall, and
    F1-score.
-   Improve the background-noise class with more varied examples.
-   Explore shorter overlapping audio windows for faster detection.
-   Add a carefully validated unknown / low-confidence result.
-   Improve the GUI with recording status and clearer event history.

## Acknowledgements

-   [ESC-50 Environmental Sound Classification
    Dataset](https://github.com/karolpiczak/ESC-50)
-   Python libraries: NumPy, SciPy, librosa, soundfile, scikit-learn,
    sounddevice, joblib, and Tkinter

------------------------------------------------------------------------

**Project type:** Academic / learning prototype\
**Focus areas:** Digital Signal Processing, audio feature extraction,
machine learning, Python, and GUI development
