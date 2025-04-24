"""
Usage: python der_testing.py --segments /path/to/segments --rttm "/path/to/rttm/*rttm"
"""

import argparse
import glob
from os.path import basename, splitext
import os
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate


def load_reference(segments_file):
    """
    Load ground-truth annotations from the segments file.
    Each line should be formatted as:
    
        speaker_id audio_id start_time end_time

    For each line, an annotation is created spanning from start_time to end_time.
    """
    reference = {}
    with open(segments_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                print(f"Skipping line (unexpected number of columns): {line.strip()}")
                continue
            speaker, audio_id, start_str, end_str = parts
            speaker = speaker.split("/")[0]
            try:
                start = float(start_str)
                end = float(end_str)
            except ValueError:
                print(f"Invalid time values in line: {line.strip()}")
                continue

            # Create or update the annotation for this audio file.
            ann = reference.get(audio_id, Annotation(uri=audio_id))
            ann[Segment(start, end)] = speaker
            reference[audio_id] = ann
    return reference


def load_rttm(file_path):
    """
    Load a single RTTM file into an Annotation object.
    We assume RTTM files follow the standard format where each line is:
    
        SPEAKER <audio_id> <channel> <start> <duration> <NA> <NA> <NA> <speaker>
    
    For simplicity, we set the annotation URI to the filename (without extension)
    so that it can be matched with the reference.
    """
    # Use the file name (without extension) as the audio id.
    audio_id = splitext(basename(file_path))[0]
    ann = Annotation(uri=audio_id)
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != "SPEAKER":
                continue
            try:
                start = float(parts[3])
                duration = float(parts[4])
            except ValueError:
                print(f"Invalid timing values in line: {line}")
                continue
            end = start + duration
            speaker = parts[7]
            ann[Segment(start, end)] = speaker
    return ann


def load_hypotheses(rttm_pattern):
    """
    Load all RTTM files that match the given glob pattern.
    Returns a dictionary mapping audio_id to its annotation.
    """
    hypothesis = {}
    for file_path in glob.glob(rttm_pattern):
        ann = load_rttm(file_path)
        hypothesis[ann.uri] = ann
    return hypothesis


def main():

    rttm = "EEND/rttms/VoxCeleb/Mspk_Mspk/*rttm"
    output_file = f"EEND/result/VoxCeleb/Mspk_Mspk.txt"

    segments = "diarization_dataset/unique_speaker_dataset/merged_test/VoxCeleb/Mspk/details/segments"


    # Load ground-truth annotations.
    reference = load_reference(segments)
    if not reference:
        print("No valid reference annotations were loaded.")
        return

    # Load system output (RTTM) annotations.
    hypothesis = load_hypotheses(rttm)
    if not hypothesis:
        print("No RTTM files were found or loaded.")
        return

    # Initialize DER metric.
    der_metric = DiarizationErrorRate()

    # Compute DER for each audio and accumulate for an average.
    total_der = 0.0
    count = 0

    # for audio_id, ref_ann in reference.items():
    #     hyp_ann = hypothesis.get(audio_id)
    #     if hyp_ann is None:
    #         print(f"No hypothesis found for audio '{audio_id}'. Skipping.")
    #         continue
    #     error = der_metric(ref_ann, hyp_ann)
    #     print(f"Audio {audio_id}: DER = {error * 100:.2f}%")
    #     total_der += error
    #     count += 1

    # if count > 0:
    #     average_der = (total_der / count) * 100
    #     print(f"\nAverage DER over {count} audio file(s): {average_der:.2f}%")
    # else:
    #     print("No matching audio files between reference and hypothesis.")

    with open(output_file, "w") as f:
        total_der = 0
        count = 0

        for audio_id, ref_ann in reference.items():
            hyp_ann = hypothesis.get(audio_id)
            if hyp_ann is None:
                print(f"No hypothesis found for audio '{audio_id}'. Skipping.")
                continue
            error = der_metric(ref_ann, hyp_ann)
            der_percentage = error * 100
            f.write(f"Audio {audio_id}: DER = {der_percentage:.2f}%\n")
            print(f"Audio {audio_id}: DER = {der_percentage:.2f}%")

            total_der += error
            count += 1

        if count > 0:
            average_der = (total_der / count) * 100
            f.write(f"\nAverage DER over {count} audio file(s): {average_der:.2f}%\n")
            print(f"\nAverage DER over {count} audio file(s): {average_der:.2f}%")
        else:
            f.write("No matching audio files between reference and hypothesis.\n")
            print("No matching audio files between reference and hypothesis.")


if __name__ == "__main__":
    main()
