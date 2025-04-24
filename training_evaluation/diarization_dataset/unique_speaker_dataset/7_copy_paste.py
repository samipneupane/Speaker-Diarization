import shutil
import os

def copy_file(source_path, destination_path):
    try:
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"The source file '{source_path}' does not exist.")
        
        dest_dir = os.path.dirname(destination_path)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        shutil.copy(source_path, destination_path)
        return f"File successfully copied to '{destination_path}'."
    
    except Exception as e:
        return f"An error occurred: {e}"


type_of_merge = "merged_audio_Mspk"



type_datasets = ['train', 'valid']
file_names = ['utt2spk', 'spk2utt']

for type_dataset in type_datasets:
    for file_name in file_names:
        source = f"unique_speaker_dataset/mixed_unique/{type_dataset}_kaldi/{file_name}"
        destination = f"unique_speaker_dataset/{type_of_merge}/{type_dataset}/details/{file_name}"
        print(copy_file(source, destination))
