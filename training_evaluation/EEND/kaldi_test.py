import os
import soundfile as sf

def generate_wav_scp(audio_dir, output_file):
    with open(output_file, 'w') as f:
        for subdir, _, files in os.walk(audio_dir):
            for file in files:
                if file.endswith('.flac') or file.endswith('.wav'):
                    utt_id = file.split('.')[0]
                    file_path = os.path.join(subdir, file).replace("\\", "/")
                    f.write(f'{utt_id} {file_path}\n')


def generate_utt2spk(audio_dir, output_file):
    with open(output_file, 'w') as f:
        for subdir, _, files in os.walk(audio_dir):
            speaker_id = os.path.basename(subdir)
            for file in files:
                if file.endswith('.flac') or file.endswith('.wav'):
                    utt_id = file.split('.')[0]
                    f.write(f'{utt_id} {speaker_id}\n')


def generate_segments(wav_scp_file, output_file):
    with open(wav_scp_file, 'r') as wav_scp, open(output_file, 'w') as segments_file:
        for line in wav_scp:
            utt_id, audio_path = line.strip().split()
            
            audio_data, samplerate = sf.read(audio_path)
            duration = len(audio_data) / samplerate
            
            start_time = 0.00
            end_time = round(duration, 2)
            
            recording_id = utt_id
            segments_file.write(f'{utt_id} {recording_id} {start_time:.2f} {end_time:.2f}\n')


#usage
which_dataset = 'diarization_dataset/already_merged_dataset/RAMC/_test'

os.makedirs(f'{which_dataset}/test_details', exist_ok=True)

generate_wav_scp(f'{which_dataset}/audio', f'{which_dataset}/test_details/wav.scp')
generate_utt2spk(f'{which_dataset}/audio', f'{which_dataset}/test_details/utt2spk')
generate_segments(f'{which_dataset}/test_details/wav.scp', f'{which_dataset}/test_details/segments')