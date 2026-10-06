from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------
# Settings
# -----------------------------
SEQUENCE_LENGTH = 50
HIDDEN_SIZE = 128
NUM_LAYERS = 1
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

DATA_FILE = Path("data/processed/notes.pkl")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# LSTM Model
# -----------------------------
class MusicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.fc(output[:, -1, :])
        return output


# -----------------------------
# Load processed data
# -----------------------------
print("Loading processed data...")

with open(DATA_FILE, "rb") as f:
    data = pickle.load(f)

encoded_notes = data["encoded_notes"]
note_to_int = data["note_to_int"]

num_notes = len(note_to_int)

print(f"Total encoded notes: {len(encoded_notes)}")
print(f"Unique notes: {num_notes}")


# -----------------------------
# Create training sequences
# -----------------------------
print("Creating training sequences...")

X = []
y = []

for i in range(len(encoded_notes) - SEQUENCE_LENGTH):
    sequence = encoded_notes[i:i + SEQUENCE_LENGTH]
    target = encoded_notes[i + SEQUENCE_LENGTH]

    X.append(sequence)
    y.append(target)

X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)

print(f"Training samples: {len(X)}")


# -----------------------------
# Train/validation split
# -----------------------------
split = int(0.9 * len(X))

X_train = X[:split]
y_train = y[:split]

X_val = X[split:]
y_val = y[split:]

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")


# -----------------------------
# DataLoader
# -----------------------------
train_dataset = TensorDataset(X_train, y_train)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# -----------------------------
# Model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

model = MusicLSTM(
    input_size=num_notes,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    output_size=num_notes
).to(device)


# -----------------------------
# Loss and optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# -----------------------------
# Training
# -----------------------------
training_losses = []
validation_losses = []

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for sequences, targets in train_loader:

        sequences = sequences.to(device)
        targets = targets.to(device)

        # One-hot encode the notes
        inputs = torch.nn.functional.one_hot(
            sequences,
            num_classes=num_notes
        ).float()

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_train_loss = total_loss / len(train_loader)

    # -------------------------
    # Validation
    # -------------------------
    model.eval()

    with torch.no_grad():

        val_sequences = X_val.to(device)
        val_targets = y_val.to(device)

        val_inputs = torch.nn.functional.one_hot(
            val_sequences,
            num_classes=num_notes
        ).float()

        val_outputs = model(val_inputs)

        val_loss = criterion(
            val_outputs,
            val_targets
        ).item()

    training_losses.append(average_train_loss)
    validation_losses.append(val_loss)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Train Loss: {average_train_loss:.4f} "
        f"- Validation Loss: {val_loss:.4f}"
    )


# -----------------------------
# Save model
# -----------------------------
model_path = MODEL_DIR / "music_lstm.pth"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "note_to_int": note_to_int,
        "sequence_length": SEQUENCE_LENGTH,
        "num_notes": num_notes,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
    },
    model_path
)

print(f"\nModel saved to: {model_path}")


# -----------------------------
# Save training graph
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(training_losses, label="Training Loss")
plt.plot(validation_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("LSTM Training and Validation Loss")
plt.legend()
plt.tight_layout()

graph_path = RESULTS_DIR / "training_loss.png"

plt.savefig(graph_path)
plt.close()

print(f"Training graph saved to: {graph_path}")