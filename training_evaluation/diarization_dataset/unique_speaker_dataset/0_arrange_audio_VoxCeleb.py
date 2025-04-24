import os
import shutil
import sys

def process_parent_folder(original_parent, dest_parent):
    # List all immediate subdirectories of the original parent folder.
    child_dirs = [os.path.join(original_parent, d) for d in os.listdir(original_parent)
                  if os.path.isdir(os.path.join(original_parent, d))]

    if not child_dirs:
        print(f"No subdirectories found in {original_parent}. Skipping...")
        return

    best_dir = None
    max_file_count = -1

    # Identify the child folder with the largest number of files.
    for child in child_dirs:
        file_count = 0
        for item in os.listdir(child):
            item_path = os.path.join(child, item)
            if os.path.isfile(item_path):
                file_count += 1
        # print(f"Folder '{child}' contains {file_count} files.")
        if file_count > max_file_count:
            max_file_count = file_count
            best_dir = child

    if best_dir is None or max_file_count == 0:
        print(f"No files found in any subdirectory of {original_parent}.")
        return

    print(f"Selected folder '{best_dir}' with {max_file_count} files for '{original_parent}'.")

    # Ensure the destination folder for this parent exists.
    if not os.path.exists(dest_parent):
        os.makedirs(dest_parent)

    # Copy files from the best directory to the destination folder.
    for item in os.listdir(best_dir):
        src = os.path.join(best_dir, item)
        if os.path.isfile(src):
            dst = os.path.join(dest_parent, item)
            # Handle potential filename collisions.
            if os.path.exists(dst):
                base, ext = os.path.splitext(item)
                counter = 1
                new_name = f"{base}_copy{ext}"
                dst = os.path.join(dest_parent, new_name)
                while os.path.exists(dst):
                    new_name = f"{base}_copy{counter}{ext}"
                    dst = os.path.join(dest_parent, new_name)
                    counter += 1
            shutil.copy2(src, dst)
            # print(f"Copied '{src}' to '{dst}'.")

def main(root_folder):
    """Process each folder in the root directory and copy files to the new structure."""
    if not os.path.isdir(root_folder):
        print(f"Error: '{root_folder}' is not a valid directory.")
        return

    # Create the result folder alongside the original root folder.
    parent_dir = os.path.dirname(os.path.abspath(root_folder))
    result_folder = "unique_speaker_dataset/VoxCeleb/arranged_audio_VoxCeleb"

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        print(f"Created result folder: {result_folder}")

    # For each subdirectory in the original root, process it.
    for item in os.listdir(root_folder):
        original_path = os.path.join(root_folder, item)
        if os.path.isdir(original_path):
            print(f"\nProcessing folder '{original_path}'...")
            dest_path = os.path.join(result_folder, item)
            process_parent_folder(original_path, dest_path)
    print("\nAll processing complete.")


if __name__ == "__main__":
    root_dir = "unique_speaker_dataset/VoxCeleb/raw_audio"
    main(root_dir)

