import os

def merge_kaldi_files(input_dirs, output_dir, filenames):
    """
    Merge corresponding Kaldi files from multiple directories into one.
    
    Parameters:
        input_dirs (list): List of folder paths containing the Kaldi files.
        output_dir (str): Path to the output directory where merged files will be saved.
        filenames (list): List of filenames to merge (e.g. ['segments', 'wav.scp', ...]).
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for fname in filenames:
        output_file_path = os.path.join(output_dir, fname)
        with open(output_file_path, 'w') as out_file:
            # Iterate over each input directory and merge the file contents
            for dir_path in input_dirs:
                input_file_path = os.path.join(dir_path, fname)
                if os.path.exists(input_file_path):
                    with open(input_file_path, 'r') as in_file:
                        # Read and write the file content
                        content = in_file.read()
                        out_file.write(content)
                        # Ensure there's a newline between files if needed
                        if content and not content.endswith('\n'):
                            out_file.write('\n')
                else:
                    print(f"Warning: {input_file_path} does not exist.")
        print(f"Merged file created: {output_file_path}")



if __name__ == "__main__":

    type_datasets = ["train", "valid"]
    speaker_nos = ["2", "M"]

    for type_dataset in type_datasets:
        for speaker_no in speaker_nos:
            # Define the input directories

            input_dirs = [
                f"unique_speaker_dataset/merged_audio_{speaker_no}spk/{type_dataset}/details",
                f"already_merged_dataset/RAMC/{type_dataset}/details"
            ]

            if speaker_no == "M":
                input_dirs.append(f"already_merged_dataset/CallHome/split_audio/{type_dataset}/details")
            
            # Directory where the merged output will be stored
            output_dir = f"merged_details/combined_{speaker_no}spk/{type_dataset}_details"
            
            # List of Kaldi file names to merge
            filenames = ["segments", "wav.scp", "reco2dur", "spk2utt", "utt2spk"]
            
            merge_kaldi_files(input_dirs, output_dir, filenames)
