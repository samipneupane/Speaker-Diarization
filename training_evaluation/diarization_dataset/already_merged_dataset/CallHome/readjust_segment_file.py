input_file = "segments"  # Update with your file name
output_file = "fixed_segments"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        parts = line.strip().split()
        if len(parts) == 4:
            utt_id, rec_id, start, duration = parts
            start = float(start)
            duration = float(duration)
            end = start + duration
            outfile.write(f"{utt_id} {rec_id} {start:.2f} {end:.2f}\n")

print(f"Fixed segments saved to {output_file}")
