import os
import shutil

def merge_kaldi_files(test_dir, merged_details_dir, kaldi_files):
    # Create merged details directory if it doesn't exist
    os.makedirs(merged_details_dir, exist_ok=True)

    # Initialize file handles for each Kaldi file to write merged content.
    file_handles = {}
    try:
        for fname in kaldi_files:
            merged_file_path = os.path.join(merged_details_dir, fname)
            file_handles[fname] = open(merged_file_path, 'w', encoding='utf-8')

        # Iterate through each subdirectory in the test directory.
        for entry in os.listdir(test_dir):
            subdir = os.path.join(test_dir, entry)
            if os.path.isdir(subdir):
                # For each expected Kaldi file, if it exists in the subdirectory, append its content.
                for fname in kaldi_files:
                    file_path = os.path.join(subdir, fname)
                    if os.path.isfile(file_path):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            # Write the content into the merged file
                            for line in f:
                                file_handles[fname].write(line)
    finally:
        # Make sure all file handles are closed.
        for handle in file_handles.values():
            handle.close()

def merge_audio_files(test_dir, merged_audio_dir):
    # Create merged audio directory if it doesn't exist
    os.makedirs(merged_audio_dir, exist_ok=True)

    # Iterate through each subdirectory in the test directory.
    for entry in os.listdir(test_dir):
        subdir = os.path.join(test_dir, entry)
        if os.path.isdir(subdir):
            audio_files_dir = os.path.join(subdir, 'audio_files')
            if os.path.isdir(audio_files_dir):
                # Iterate over all files in the audio_files directory.
                for audio_file in os.listdir(audio_files_dir):
                    src_audio_path = os.path.join(audio_files_dir, audio_file)
                    # Ensure it is a file
                    if os.path.isfile(src_audio_path):
                        dst_audio_path = os.path.join(merged_audio_dir, audio_file)
                        # If a file with the same name exists, you may want to handle it (e.g., rename)
                        if os.path.exists(dst_audio_path):
                            # Option: Append the parent directory name to avoid collisions.
                            base, ext = os.path.splitext(audio_file)
                            dst_audio_path = os.path.join(merged_audio_dir, f"{entry}_{base}{ext}")
                        shutil.copy2(src_audio_path, dst_audio_path)

def main():
    # Base directory containing the random-named subdirectories.
    test_dir = 'valid_'
    # Directory to merge all audio files
    merged_audio_dir = 'valid/audio'
    # Directory to merge all kaldi details files
    merged_details_dir = os.path.join('valid/details')
    
    # List of Kaldi files to merge.
    kaldi_files = ['reco2dur', 'utt2spk', 'spk2utt', 'segments', 'wav.scp']
    
    # Merge audio files.
    merge_audio_files(test_dir, merged_audio_dir)
    print(f"All audio files have been copied to '{merged_audio_dir}'.")

    # Merge kaldi files.
    merge_kaldi_files(test_dir, merged_details_dir, kaldi_files)
    print(f"All Kaldi files have been merged into '{merged_details_dir}'.")

if __name__ == '__main__':
    main()
