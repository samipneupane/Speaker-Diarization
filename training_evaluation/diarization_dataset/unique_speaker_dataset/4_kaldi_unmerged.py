import os
import soundfile as sf
from collections import defaultdict

def generate_utt2spk(root_dir, output_file):
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, 'w') as f:
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.flac') or file.endswith('.wav') or file.endswith('.m4a') or file.endswith('.mp3'):
                    # make utt_id from subdir separeted by '/'
                    utt_id = subdir.split('/')[-1]
                    file_path = os.path.join(subdir, file)
                    file_path = file_path.replace("\\", "/")

                    parent_dir, file_name = os.path.split(file_path)
                    utterance = os.path.join(os.path.basename(parent_dir), os.path.splitext(file_name)[0])

                    f.write(f'{utterance} {utt_id}\n')


def convert_utt2spk_to_spk2utt(input_file, output_file):
    spk2utt = defaultdict(list)
    
    # Read the utt2spk file and organize data by speaker
    with open(input_file, 'r') as f:
        for line in f:
            utt, spk = line.strip().split()
            spk2utt[spk].append(utt)
    
    # Write the spk2utt file
    with open(output_file, 'w') as f:
        for spk, utts in spk2utt.items():
            f.write(f"{spk} {' '.join(utts)}\n")



type_dataset = "valid"

audio_path = f"unique_speaker_dataset/mixed_unique/{type_dataset}"
utt2spk_path = f"unique_speaker_dataset/mixed_unique/{type_dataset}_kaldi/utt2spk"
generate_utt2spk(audio_path, utt2spk_path)

input_file = f"unique_speaker_dataset/mixed_unique/{type_dataset}_kaldi/utt2spk"
output_file = f"unique_speaker_dataset/mixed_unique/{type_dataset}_kaldi/spk2utt"
convert_utt2spk_to_spk2utt(input_file, output_file)
