#!/usr/bin/env python3
import os
import wave
import contextlib
import sys

# Default audio directory (no CLI args needed)
AUDIO_DIR = 'diarization_dataset/unique_speaker_dataset/merged_test/VoxCeleb/Mspk/audio'


def get_wav_duration(path):
    """
    Returns the duration of a WAV file in seconds.
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)


def total_audio_length_hours(root_dir, extensions=('.wav',)):
    """
    Walks through root_dir, finds files ending with any of the given extensions,
    sums their durations, and returns total length in hours.
    """
    total_seconds = 0.0
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(extensions):
                full_path = os.path.join(dirpath, fname)
                try:
                    total_seconds += get_wav_duration(full_path)
                except wave.Error as e:
                    print(f"Warning: could not read '{full_path}': {e}", file=sys.stderr)
                except Exception as e:
                    print(f"Error processing '{full_path}': {e}", file=sys.stderr)
    return total_seconds / 3600.0


def main():
    # Use the default AUDIO_DIR
    if not os.path.isdir(AUDIO_DIR):
        print(f"Error: Default audio directory '{AUDIO_DIR}' does not exist.", file=sys.stderr)
        sys.exit(1)

    total_hours = total_audio_length_hours(AUDIO_DIR)
    print(f"Total audio length under '{AUDIO_DIR}': {total_hours:.2f} hours")


if __name__ == "__main__":
    main()
