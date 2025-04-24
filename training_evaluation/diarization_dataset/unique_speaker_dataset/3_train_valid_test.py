import os
import shutil
import random
import math


def train_valid_test_split(audio_dir, train_per, valid_per):

    # Get all person directories in the audio folder
    person_dirs = [d for d in os.listdir(audio_dir) if os.path.isdir(os.path.join(audio_dir, d))]

    # Shuffle the directories for random splitting
    random.shuffle(person_dirs)

    # Calculate the split sizes
    total = len(person_dirs)
    train_size = math.floor(total * train_per)
    validation_size = math.floor(total * valid_per)
    test_size = total - train_size - validation_size

    # Split the directories
    train_dirs = person_dirs[:train_size]
    validation_dirs = person_dirs[train_size:train_size + validation_size]
    test_dirs = person_dirs[train_size + validation_size:]

    # Define train, validation, and test directories
    train_dir = os.path.join(audio_dir, "train")
    validation_dir = os.path.join(audio_dir, "valid")
    test_dir = os.path.join(audio_dir, "_test")

    # Create the directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Move directories to their respective folders
    for person in train_dirs:
        shutil.move(os.path.join(audio_dir, person), train_dir)

    for person in validation_dirs:
        shutil.move(os.path.join(audio_dir, person), validation_dir)

    for person in test_dirs:
        shutil.move(os.path.join(audio_dir, person), test_dir)

    print(f"Moved {len(train_dirs)} directories to {train_dir}")
    print(f"Moved {len(validation_dirs)} directories to {validation_dir}")
    print(f"Moved {len(test_dirs)} directories to {test_dir}")



if __name__ == "__main__":
    # modify train_per and valid_per
    train_per = 0.6
    valid_per = 0.2 
    # test_per is automatically 1.0-(train_per+valid_per)

    audio_dir = "unique_speaker_dataset/LibriSpeech/VA_LibriSpeech"
    train_valid_test_split(audio_dir, train_per, valid_per)

    audio_dir = "unique_speaker_dataset/NeHi/VA_NeHi"
    train_valid_test_split(audio_dir, train_per, valid_per)

    audio_dir = "unique_speaker_dataset/VoxCeleb/arranged_audio_VoxCeleb"
    train_valid_test_split(audio_dir, train_per, valid_per)