from pathlib import Path
import pickle
from music21 import converter, note, chord


# Project folders
DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def extract_notes(midi_file):
    """Extract note names from one MIDI file."""
    score = converter.parse(midi_file)

    notes = []

    for element in score.flatten().notes:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            # Use the lowest note for chords
            notes.append(str(element.pitches[0]))

    return notes


def main():
    midi_files = list(DATA_DIR.rglob("*.midi"))

    print(f"MIDI files found: {len(midi_files)}")

    all_notes = []

    for i, midi_file in enumerate(midi_files, start=1):
        print(f"Processing {i}/{len(midi_files)}: {midi_file.name}")

        try:
            notes = extract_notes(midi_file)
            all_notes.extend(notes)
        except Exception as e:
            print(f"Could not process {midi_file.name}: {e}")

    print(f"\nTotal notes extracted: {len(all_notes)}")

    # Create vocabulary
    unique_notes = sorted(set(all_notes))

    note_to_int = {
        note_name: i
        for i, note_name in enumerate(unique_notes)
    }

    int_to_note = {
        i: note_name
        for note_name, i in note_to_int.items()
    }

    # Convert notes to integers
    encoded_notes = [note_to_int[n] for n in all_notes]

    # Save processed data
    output_file = PROCESSED_DIR / "notes.pkl"

    with open(output_file, "wb") as f:
        pickle.dump(
            {
                "notes": all_notes,
                "encoded_notes": encoded_notes,
                "note_to_int": note_to_int,
                "int_to_note": int_to_note,
            },
            f,
        )

    print(f"Unique notes: {len(unique_notes)}")
    print(f"Processed data saved to: {output_file}")


if __name__ == "__main__":
    main()