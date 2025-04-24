import os
import soundfile as sf

type_dir = "valid"
audio_dir = f"unique_speaker_dataset/merged_audio_Mspk/{type_dir}/audio"
reco2dur_path = f"unique_speaker_dataset/merged_audio_Mspk/{type_dir}/details/reco2dur"

def get_duration(file_path):
    with sf.SoundFile(file_path) as audio:
        return len(audio) / audio.samplerate

def create_reco2dur(audio_dir, reco2dur_path):
    flac_files = [os.path.join(audio_dir, file) for file in os.listdir(audio_dir) if file.endswith('.wav')]
    
    with open(reco2dur_path, 'w') as f:
        for file_path in flac_files:
            duration = get_duration(file_path)
            filename = os.path.splitext(os.path.basename(file_path))[0]
            f.write(f"{filename} {duration}\n")

if __name__ == "__main__":
    create_reco2dur(audio_dir, reco2dur_path)
