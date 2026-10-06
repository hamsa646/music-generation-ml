from pathlib import Path
import pickle

import torch
import torch.nn as nn
from music21 import note, stream


# -----------------------------
# Settings
# -----------------------------
MODEL_FILE = Path("models/music_lstm.pth")
OUTPUT_DIR = Path("generated")

NUM_NOTES_TO_GENERATE = 200
TEMPERATURE = 1.0


# -----------------------------
# LSTM Model
# -----------------------------
class MusicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])


# -----------------------------
# Load trained model
# -----------------------------
print("Loading trained model...")

checkpoint = torch.load(
    MODEL_FILE,
    map_location="cpu",
    weights_only=False
)

note_to_int = checkpoint["note_to_int"]
sequence_length = checkpoint["sequence_length"]
num_notes = checkpoint["num_notes"]
hidden_size = checkpoint["hidden_size"]
num_layers = checkpoint["num_layers"]

int_to_note = {
    value: key
    for key, value in note_to_int.items()
}


model = MusicLSTM(
    input_size=num_notes,
    hidden_size=hidden_size,
    num_layers=num_layers,
    output_size=num_notes
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model loaded successfully.")


# -----------------------------
# Load processed notes
# -----------------------------
with open("data/processed/notes.pkl", "rb") as f:
    data = pickle.load(f)

encoded_notes = data["encoded_notes"]


# -----------------------------
# Choose seed sequence
# -----------------------------
seed = encoded_notes[:sequence_length]

generated_notes = list(seed)


# -----------------------------
# Generate new notes
# -----------------------------
print(f"Generating {NUM_NOTES_TO_GENERATE} new notes...")

with torch.no_grad():

    for _ in range(NUM_NOTES_TO_GENERATE):

        sequence = generated_notes[-sequence_length:]

        sequence_tensor = torch.tensor(
            sequence,
            dtype=torch.long
        ).unsqueeze(0)

        inputs = torch.nn.functional.one_hot(
            sequence_tensor,
            num_classes=num_notes
        ).float()

        output = model(inputs)

        # Temperature controls randomness
        probabilities = torch.softmax(
            output / TEMPERATURE,
            dim=1
        )

        next_note = torch.multinomial(
            probabilities,
            num_samples=1
        ).item()

        generated_notes.append(next_note)


# -----------------------------
# Convert generated notes
# -----------------------------
new_notes = generated_notes[sequence_length:]

music_stream = stream.Stream()

for note_number in new_notes:

    note_name = int_to_note[note_number]

    new_note = note.Note(note_name)

    new_note.quarterLength = 0.5

    music_stream.append(new_note)


# -----------------------------
# Save MIDI
# -----------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output_file = OUTPUT_DIR / "generated_song.mid"

music_stream.write(
    "midi",
    fp=output_file
)

print(f"Generated MIDI saved to: {output_file}")