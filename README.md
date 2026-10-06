# Generating Music with Machine Learning

## Project Overview
This mini-project implements a simplified LSTM-based music generation system inspired by the paper **“Project milestone: Generating music with Machine Learning”**.

The system learns note-sequence patterns from MIDI files and generates a new sequence of notes using an LSTM recurrent neural network.

## Problem Statement
Given a sequence of musical notes from MIDI files, train a machine learning model to predict the next note and use repeated predictions to generate a new musical sequence.

## Dataset
The project uses MIDI files from the **MAESTRO v3.0.0** dataset. For the implementation, 20 MIDI files were used.

- MIDI files used: 20
- Total extracted notes: 82,157
- Unique note classes: 88

The MIDI files are not included in this repository. Download the MIDI-only MAESTRO dataset and place the selected `.midi` files in:

```text
data/raw/
```

## Methodology

```text
MIDI Files
   ↓
Preprocessing with music21
   ↓
Note extraction and integer encoding
   ↓
Sequences of 50 notes
   ↓
LSTM Neural Network
   ↓
Next-note prediction
   ↓
200 generated notes
   ↓
Generated MIDI file
```

## Model
The implementation uses a single-layer LSTM.

| Parameter | Value |
|---|---:|
| Sequence length | 50 |
| Hidden size | 128 |
| LSTM layers | 1 |
| Output classes | 88 |
| Batch size | 64 |
| Epochs | 10 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss | Cross Entropy |

## Project Structure

```text
Music-Generation-ML/
├── data/
│   ├── raw/
│   └── processed/
├── generated/
├── models/
├── results/
│   └── training_loss.png
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── generate.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/hamsa646/music-generation-ml.git
cd music-generation-ml
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate it on Windows

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

If PyTorch installation fails because of the CPU build, install the CPU version separately using the official PyTorch installation command and then install the remaining packages.

## Running the Project

### Step 1 — Add MIDI files

Place `.midi` files inside:

```text
data/raw/
```

### Step 2 — Preprocess

```bash
python src/preprocess.py
```

This creates:

```text
data/processed/notes.pkl
```

### Step 3 — Train

```bash
python src/train.py
```

This creates:

```text
models/music_lstm.pth
results/training_loss.png
```

### Step 4 — Generate music

```bash
python src/generate.py
```

This creates:

```text
generated/generated_song.mid
```

Open the generated MIDI file using MuseScore Studio to listen to the generated sequence.

## Results

Training loss decreased from **3.7641** in epoch 1 to **3.0868** in epoch 10.

Validation loss was **3.5570** in epoch 1 and **3.4109** in epoch 10.

The model successfully generated a playable MIDI file containing 200 generated notes.

## Limitations

- The implementation uses only note pitch information.
- Note duration and velocity are simplified/not fully modeled.
- The dataset subset is small compared with the complete MAESTRO dataset.
- The project focuses on a simple LSTM pipeline rather than implementing all approaches from the reference paper.
- Human evaluation was not performed in this implementation.

## Future Work

- Add note duration and velocity features.
- Use a larger MIDI dataset.
- Compare LSTM with GRU or Transformer architectures.
- Add quantitative next-note prediction metrics.
- Perform human evaluation of generated music.

## Conclusion

The project demonstrates a complete machine-learning pipeline for symbolic music generation: MIDI preprocessing, sequence creation, LSTM training, next-note prediction, and MIDI generation. The generated output can be opened and played in MuseScore Studio.
