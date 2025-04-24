import os
import webrtcvad
from pydub import AudioSegment
import soundfile as sf
import numpy as np


def read_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(16000)  # Set to 16 kHz (as required by WebRTC VAD)
    audio = audio.set_channels(1)  # Convert to mono
    return audio


def write_audio(file_path, audio_data, frame_rate):
    sf.write(file_path, audio_data, frame_rate, format='FLAC')


def frame_generator(audio, frame_duration_ms=30):
    frame_size = int(audio.frame_rate * (frame_duration_ms / 1000.0) * 2)  # 16-bit PCM = 2 bytes per sample
    for i in range(0, len(audio.raw_data), frame_size):
        yield audio.raw_data[i:i + frame_size]


def vad_audio(input_file_path, output_file_path):
    vad = webrtcvad.Vad()
    vad.set_mode(3)  # Most aggressive mode

    audio = read_audio(input_file_path)
    frame_rate = audio.frame_rate

    voice_frames = []
    for frame in frame_generator(audio):
        if len(frame) == int(frame_rate * 30 / 1000) * 2:  # Ensure frame is exactly 30ms
            is_speech = vad.is_speech(frame, frame_rate)
            if is_speech:
                voice_frames.append(frame)

    if voice_frames:
        voice_audio = b"".join(voice_frames)
        audio_np = np.frombuffer(voice_audio, dtype=np.int16)
        write_audio(output_file_path, audio_np, frame_rate)
        print(f"Processed and saved: {output_file_path}")
    else:
        print(f"No voice activity detected in: {input_file_path}")


def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav') or file.endswith('.flac') or file.endswith('.mp3') or file.endswith('.m4a'):
                input_file_path = os.path.join(root, file)
                
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                
                output_file_path = os.path.join(output_subdir, os.path.splitext(file)[0] + ".flac")
                vad_audio(input_file_path, output_file_path)



if __name__ == "__main__":
    input_directory = "unique_speaker_dataset/NeHi/raw_audio"
    output_directory = "unique_speaker_dataset/NeHi/VA_NeHi"
    process_directory(input_directory, output_directory)