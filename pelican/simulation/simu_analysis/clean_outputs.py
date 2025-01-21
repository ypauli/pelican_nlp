import os
import shutil
import json

def remove_a(base_dir):
    metadata_dir = os.path.join(base_dir, "Metadata")
    # Iterate over all subject folders in the base directory
    for subject in os.listdir(metadata_dir):
        subject_path = os.path.join(metadata_dir, subject)

        # Check if the path is a directory
        if os.path.isdir(subject_path):
            inner_a_path = os.path.join(subject_path, "a")
            
            # Check if the 'a' directory exists
            if os.path.exists(inner_a_path) and os.path.isdir(inner_a_path):
                # Rename the middle 'a' folder to avoid conflicts
                temp_path = os.path.join(subject_path, "temp_a")
                os.rename(inner_a_path, temp_path)

                # Move all files/folders inside 'temp_a' to the subject directory
                for item in os.listdir(temp_path):
                    item_path = os.path.join(temp_path, item)
                    shutil.move(item_path, subject_path)
                
                # Remove the now empty 'temp_a' directory
                os.rmdir(temp_path)

    print("Files moved and 'a' directories renamed and removed.")


def rename_subjects(base_path, start_old, end_old, start_new):
    metadata_path = os.path.join(base_path, "Metadata")
    subjects_path = os.path.join(base_path, "Subjects")

    old_subjects = [f"subject_{i}" for i in range(start_old, end_old + 1)]
    new_subjects = [f"subject_{i}" for i in range(start_new, start_new + (end_old - start_old) + 1)]

    # Create a mapping between old and new names
    mapping = dict(zip(old_subjects, new_subjects))

    # Rename in Metadata folder
    for old_subject, new_subject in mapping.items():
        old_metadata_path = os.path.join(metadata_path, old_subject)
        new_metadata_path = os.path.join(metadata_path, new_subject)

        if os.path.exists(old_metadata_path):
            shutil.move(old_metadata_path, new_metadata_path)
            print(f"Renamed {old_metadata_path} to {new_metadata_path}")

    # Rename in Subjects folder
    for old_subject, new_subject in mapping.items():
        old_subject_path = os.path.join(subjects_path, old_subject)
        new_subject_path = os.path.join(subjects_path, new_subject)

        if os.path.exists(old_subject_path):
            # Rename the main subject folder
            shutil.move(old_subject_path, new_subject_path)
            print(f"Renamed {old_subject_path} to {new_subject_path}")

            # Rename files inside the session folder(s)
            for ses_folder in os.listdir(new_subject_path):
                ses_path = os.path.join(new_subject_path, ses_folder)

                if os.path.isdir(ses_path):
                    for root, _, files in os.walk(ses_path):
                        for file in files:
                            # Replace old subject ID with new subject ID in filenames
                            old_filename = os.path.join(root, file)
                            new_filename = old_filename.replace(f"sub-{old_subject.split('_')[1]}", f"sub-{new_subject.split('_')[1]}")

                            if old_filename != new_filename:
                                os.rename(old_filename, new_filename)
                                print(f"Renamed file {old_filename} to {new_filename}")
    
def update_metadata_to_match_subject(base_path):
    base_path = os.path.join(base_path, "Metadata")
    # Traverse subject_x directories
    for subject_dir in os.listdir(base_path):
        subject_path = os.path.join(base_path, subject_dir)
        
        # Ensure it's a directory and follows the "subject_x" format
        if os.path.isdir(subject_path) and subject_dir.startswith("subject_"):
            try:
                subject_number = int(subject_dir.split("_")[1])  # Extract subject number
            except (IndexError, ValueError):
                print(f"Skipping invalid directory name: {subject_dir}")
                continue
            
            # Traverse y directories inside subject_x
            for sub_dir in os.listdir(subject_path):
                metadata_file_path = os.path.join(subject_path, sub_dir, "metadata.json")
                
                # Check if metadata.json exists
                if os.path.isfile(metadata_file_path):
                    try:
                        # Read the existing metadata.json
                        with open(metadata_file_path, "r") as f:
                            metadata = json.load(f)

                        # Update the "subject" field
                        metadata["subject"] = subject_number

                        # Write back the updated metadata
                        with open(metadata_file_path, "w") as f:
                            json.dump(metadata, f, indent=4)

                        print(f"Updated {metadata_file_path} with subject: {subject_number}")

                    except Exception as e:
                        print(f"Error updating {metadata_file_path}: {e}")
                else:
                    print(f"Metadata file not found: {metadata_file_path}")

    
# Example usage
base_dir = "/home/ubuntu/PELICAN/pelican/simulation/simu_output"
start_old = 0
end_old = 9
start_new = 0

# rename_subjects(base_dir, start_old, end_old, start_new)
# remove_a(base_dir)
# update_metadata_to_match_subject(base_dir)

# print("Directories c, d, e have been moved to a, and folder b has been removed.")

# Remove ses-0 and move all contents to parent folder on output_dir
# for subject in os.listdir(output_dir):
#     sub_dir = os.path.join(output_dir, subject, "ses-0")
#     if os.path.exists(sub_dir):
#         for item in os.listdir(sub_dir):
#             shutil.move(os.path.join(sub_dir, item), os.path.join(output_dir, subject))
#         os.rmdir(sub_dir)
#         print(f"Moved contents of ses-0 to parent folder for subject {subject}.")
#     else:
#         print(f"No ses-0 folder found for subject {subject}.")



# Rename sub_dir to sub-dir

