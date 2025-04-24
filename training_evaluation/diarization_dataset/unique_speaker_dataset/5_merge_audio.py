import os
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple
from pydub import AudioSegment
import numpy as np


@dataclass
class MergeConfig:
    # [Previous config code remains the same]
    output_count: int = 1000
    min_speakers: int = 2
    max_speakers: int = 4
    min_utts_per_spk: int = 1
    max_utts_per_spk: int = 3
    silence_prob: float = 0.7
    min_silence_len: int = 500
    max_silence_len: int = 2000
    overlap_prob: float = 0.3
    min_overlap_len: int = 500
    max_overlap_len: int = 2000
    max_overlap_ratio: float = 0.5
    file_format: str = "flac"
    
    def validate(self, total_speakers: int):
        if self.min_speakers > total_speakers:
            print(f"Warning: Adjusting min_speakers from {self.min_speakers} to {total_speakers}")
            self.min_speakers = total_speakers
        
        if self.max_speakers > total_speakers:
            print(f"Warning: Adjusting max_speakers from {self.max_speakers} to {total_speakers}")
            self.max_speakers = total_speakers
        
        if self.min_speakers > self.max_speakers:
            self.min_speakers = self.max_speakers
            print(f"Warning: Adjusted min_speakers to {self.min_speakers} to match max_speakers")


def load_speaker_data(audio_dir: str, file_format: str) -> Dict[str, List[str]]:
    # [Previous load_speaker_data code remains the same]
    speaker_files = {}
    for speaker_dir in os.listdir(audio_dir):
        speaker_path = os.path.join(audio_dir, speaker_dir)
        if os.path.isdir(speaker_path):
            files = [
                os.path.join(speaker_path, f)
                for f in os.listdir(speaker_path)
                if f.endswith(file_format) or f.endswith("wav")
            ]
            if files:
                speaker_files[speaker_dir] = files
    
    if not speaker_files:
        raise ValueError(f"No audio files found in {audio_dir} with format .{file_format} or .wav")
    
    print(f"Found {len(speaker_files)} speakers with valid audio files")
    return speaker_files


def create_silence(duration_ms: int) -> AudioSegment:
    return AudioSegment.silent(duration=duration_ms)


def process_utterance(utt_path: str, config: MergeConfig) -> AudioSegment:


    audio = AudioSegment.from_file(utt_path, format=config.file_format)
    if audio.channels > 1:
        audio = audio.set_channels(1)
    return audio


def merge_audio_files(speaker_files: Dict[str, List[str]], config: MergeConfig) -> Tuple[AudioSegment, List[Tuple[str, float, float, bool]]]:
    # Select speakers and utterances
    max_available_speakers = len(speaker_files)
    num_speakers = random.randint(
        min(config.min_speakers, max_available_speakers),
        min(config.max_speakers, max_available_speakers)
    )
    
    selected_speakers = random.sample(list(speaker_files.keys()), num_speakers)
    utterances = []
    for speaker in selected_speakers:
        available_utts = len(speaker_files[speaker])
        num_utts = random.randint(
            min(config.min_utts_per_spk, available_utts),
            min(config.max_utts_per_spk, available_utts)
        )
        speaker_utts = random.sample(speaker_files[speaker], num_utts)
        utterances.extend((utt, speaker) for utt in speaker_utts)
    
    random.shuffle(utterances)
    
    merged_audio = AudioSegment.empty()
    segments_info = []
    current_time = 0.0
    
    i = 0
    while i < len(utterances):
        current_utt_path, _ = utterances[i]
        current_audio = process_utterance(current_utt_path, config)
        parent_dir, file_name = os.path.split(current_utt_path)
        current_utt_id = os.path.join(os.path.basename(parent_dir), os.path.splitext(file_name)[0])
        # current_utt_id = os.path.basename(current_utt_path).rsplit('.', 1)[0]
        
        # Add silence before utterance if needed
        if random.random() < config.silence_prob and i > 0:
            silence_duration = random.randint(config.min_silence_len, config.max_silence_len)
            merged_audio += create_silence(silence_duration)
            current_time += silence_duration / 1000
        
        # Check if we should create overlap with next utterance
        if i < len(utterances) - 1 and random.random() < config.overlap_prob:
            next_utt_path, _ = utterances[i + 1]
            next_audio = process_utterance(next_utt_path, config)
            
            # Calculate current utterance duration in seconds
            current_duration = len(current_audio) / 1000  # Convert ms to seconds
            
            # Calculate valid overlap range (in seconds)
            max_overlap = min(config.max_overlap_len / 1000,  # Convert ms to seconds
                            current_duration * config.max_overlap_ratio)
            min_overlap = config.min_overlap_len / 1000  # Convert ms to seconds
            
            if max_overlap >= min_overlap:
                # Choose a random overlap duration within valid range
                overlap_duration = random.uniform(min_overlap, max_overlap)
                
                # Calculate when the next utterance should start
                next_start_time = current_time + current_duration - overlap_duration
                
                # Add current audio
                merged_audio += current_audio
                segments_info.append((current_utt_id, current_time, 
                                   current_time + current_duration, True))
                
                # Add overlapped part of next audio
                overlap_ms = int(overlap_duration * 1000)  # Convert back to ms for pydub
                overlap_audio = next_audio[:overlap_ms]
                
                # Calculate position for overlay
                position_ms = len(merged_audio) - overlap_ms
                merged_audio = merged_audio.overlay(overlap_audio, position=position_ms)
                
                # Add remaining part of next audio
                if len(next_audio) > overlap_ms:
                    merged_audio += next_audio[overlap_ms:]
                
                # Add segment info for next utterance

                parent_dir, file_name = os.path.split(next_utt_path)
                next_utt_id = os.path.join(os.path.basename(parent_dir), os.path.splitext(file_name)[0])
                # next_utt_id = os.path.basename(next_utt_path).rsplit('.', 1)[0]
                next_duration = len(next_audio) / 1000
                segments_info.append((next_utt_id, next_start_time,
                                   next_start_time + next_duration, True))
                
                current_time = next_start_time + next_duration
                i += 2  # Skip next utterance since we've processed it
                continue
        
        # If no overlap, add current audio normally
        merged_audio += current_audio
        current_duration = len(current_audio) / 1000
        segments_info.append((current_utt_id, current_time,
                            current_time + current_duration, False))
        current_time += current_duration
        i += 1
    
    return merged_audio, segments_info


def prepare_dataset(unmerged_dir: str, merged_dir: str, config: MergeConfig):
    # [Previous prepare_dataset code remains the same]
    audio_dir = os.path.join(unmerged_dir)
    merged_audio_dir = os.path.join(merged_dir, "audio")
    details_dir = os.path.join(merged_dir, "details")
    
    os.makedirs(merged_audio_dir, exist_ok=True)
    os.makedirs(details_dir, exist_ok=True)
    
    merge_lines = []
    wav_scp_lines = []
    segments_lines = []
    
    speaker_files = load_speaker_data(audio_dir, config.file_format)
    config.validate(len(speaker_files))
    
    for i in range(config.output_count):
        merged_audio, segments_info = merge_audio_files(speaker_files, config)
        # print(segments_info)
        
        output_file = os.path.join(merged_audio_dir, f"{i:08d}.wav")
        merged_audio.export(output_file, format="wav")
        
        merged_id = f"{i:08d}"
        wav_scp_lines.append(f"{merged_id} {output_file}")
        merge_lines.append(f"{merged_id} {' '.join(seg[0] for seg in segments_info)}")
        
        for utt_id, start_time, end_time, is_overlapped in segments_info:
            segments_lines.append(f"{utt_id} {merged_id} {start_time:.2f} {end_time:.2f}")
        
        print(f"Processed file {i+1}/{config.output_count}")
    
    with open(os.path.join(details_dir, "wav.scp"), 'w') as f:
        f.write("\n".join(line.replace("\\", "/") for line in wav_scp_lines) + '\n')
    
    with open(os.path.join(details_dir, "merge"), 'w') as f:
        f.write("\n".join(merge_lines) + '\n')
    
    with open(os.path.join(details_dir, "segments"), 'w') as f:
        f.write("\n".join(segments_lines) + '\n')



if __name__ == "__main__":
    config = MergeConfig(
        output_count=2000, #####
        min_speakers=2, #####
        max_speakers=2, #####
        min_utts_per_spk=1, #####
        max_utts_per_spk=3, #####
        silence_prob=0.5, #####
        min_silence_len=500,
        max_silence_len=2000,
        overlap_prob=0.0, #####
        min_overlap_len=500,
        max_overlap_len=2000,
        max_overlap_ratio=0.5,
        file_format="flac"
    )
    
    which_dataset = "_test"
    unmerged_dir = f"unique_speaker_dataset/VoxCeleb/arranged_audio_VoxCeleb/{which_dataset}"
    merged_dir = f"unique_speaker_dataset/merged_test/VoxCeleb/2spk"
    
    prepare_dataset(unmerged_dir, merged_dir, config)