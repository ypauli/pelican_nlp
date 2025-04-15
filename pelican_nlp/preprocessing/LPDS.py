import re
import os

from pelican_nlp.config import debug_print

class LPDS:
    def __init__(self, project_folder, multiple_sessions):
        self.project_folder = project_folder
        self.multiple_sessions = multiple_sessions
        self.subjects_folder = os.path.join(self.project_folder, "subjects")
        self.subject_folders = [f for f in os.listdir(self.subjects_folder) if
                                os.path.isdir(os.path.join(self.subjects_folder, f))]

    def LPDS_checker(self):
        # Check if the main project folder exists
        if not os.path.isdir(self.project_folder):
            raise FileNotFoundError(f"Project folder '{self.project_folder}' does not exist.")

        # Check for required files in the project folder
        suggested_files = ["dataset_description.json", "README", "CHANGES", "participants.tsv"]
        for file in suggested_files:
            if not os.path.isfile(os.path.join(self.project_folder, file)):
                debug_print(f"Warning: Missing suggested file '{file}' in the project folder.")

        # Check for the 'subjects' folder
        if not os.path.isdir(self.subjects_folder):
            raise FileNotFoundError("Error: The 'subjects' folder is missing in the project folder.")

        # Check if there is at least one subfolder in 'subjects', ideally named 'sub-01'
        if not self.subject_folders:
            raise FileNotFoundError("Error: No subject subfolders found in the 'subjects' folder.")
        if 'sub-01' not in self.subject_folders:
            print("Warning: Ideally, subject folders should follow the naming convention 'sub-x'.")

        # Iterate through subject subfolders
        for subject_folder in self.subject_folders:
            subject_path = os.path.join(self.subjects_folder, subject_folder)

            # Check for session folders if project has sessions
            if self.multiple_sessions:
                session_folders = [f for f in os.listdir(subject_path) if
                                   os.path.isdir(os.path.join(subject_path, f))]
                if session_folders:
                    if 'ses-01' not in session_folders:
                        print(f"Warning: Ideally, the session folders should follow the naming convention 'ses-x'.")
                else:
                    print(f"Warning: No session folders found in '{subject_folder}'.")

            # Check for optional subject_metadata file
            metadata_file = os.path.join(subject_path, "subject_metadata")
            if not os.path.isfile(metadata_file):
                debug_print(f"Note: Optional 'subject_metadata' file is missing in '{subject_folder}'.")
                continue

            session_folders = subject_folder

            # Iterate through current level folders (subjects or sessions)
            for session_folder in session_folders:
                session_path = os.path.join(subject_path, session_folder)
                task_folders = [f for f in os.listdir(session_path) if os.path.isdir(os.path.join(session_path, f))]

                # Check for tasks inside session folder
                for task_folder in task_folders:
                    task_path = os.path.join(session_path, task_folder)
                    task_files = [f for f in os.listdir(task_path) if os.path.isfile(os.path.join(task_path, f))]

                    # Check naming convention for files in the task folder
                    for file in task_files:
                        if self.multiple_sessions:
                            pattern = fr"^{subject_folder}_{session_folder}_{task_folder}.*"
                        else:
                            pattern = fr"^{subject_folder}_{task_folder}.*"
                        if not re.match(pattern, file):
                            debug_print(f"Warning: File '{file}' in '{task_folder}' does not follow the LPDS naming conventions")

    def derivative_dir_creator(self):
        # Create the 'derivatives' folder if it doesn't exist
        derivatives_folder = os.path.join(self.project_folder, "derivatives")
        if not os.path.exists(derivatives_folder):
            os.makedirs(derivatives_folder)