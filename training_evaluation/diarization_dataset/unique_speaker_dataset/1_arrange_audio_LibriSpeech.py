import os
import shutil

def copy_audio_files(src_base, dest_base):
    # Create destination base directory if it doesn't exist
    if not os.path.exists(dest_base):
        os.makedirs(dest_base)
        print(f"Created destination base directory: {dest_base}")

    # Iterate over each ID directory in the source base
    for id_dir in os.listdir(src_base):
        src_id_path = os.path.join(src_base, id_dir)
        # Only process directories
        if os.path.isdir(src_id_path):
            # Create corresponding ID directory in destination
            dest_id_path = os.path.join(dest_base, id_dir)
            if not os.path.exists(dest_id_path):
                os.makedirs(dest_id_path)
                print(f"Created directory: {dest_id_path}")

            # Iterate over each subdirectory in the ID directory
            for subdir in os.listdir(src_id_path):
                src_subdir_path = os.path.join(src_id_path, subdir)
                if os.path.isdir(src_subdir_path):
                    # Iterate over each file in the subdirectory
                    for file_name in os.listdir(src_subdir_path):
                        if file_name.endswith(".flac") or file_name.endswith(".wav"):
                            src_file = os.path.join(src_subdir_path, file_name)
                            dest_file = os.path.join(dest_id_path, file_name)
                            shutil.copy2(src_file, dest_file)
                            # print(f"Copied: {src_file} -> {dest_file}")

if __name__ == "__main__":
    source_directory = "unique_speaker_dataset/LibriSpeech/train-clean-360"
    destination_directory = "unique_speaker_dataset/LibriSpeech/arranged_audio_LibriSpeech"
    copy_audio_files(source_directory, destination_directory)
