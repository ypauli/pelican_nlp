import re
import os

from pelican_nlp.config import debug_print

class LPDS:
    def __init__(self, project_folder, multiple_sessions):
        self.project_folder = project_folder
        self.multiple_sessions = multiple_sessions
        self.participants_folder = os.path.join(self.project_folder, "participants")
        self.participant_folders = [f for f in os.listdir(self.participants_folder) if
                                os.path.isdir(os.path.join(self.participants_folder, f))]

    def LPDS_checker(self):
        # Check if the main project folder exists
        if not os.path.isdir(self.project_folder):
            raise FileNotFoundError(f"Project folder '{self.project_folder}' does not exist.")

        # Check for required files in the project folder
        suggested_files = ["dataset_description.json", "README", "CHANGES", "participants.tsv"]
        for file in suggested_files:
            if not os.path.isfile(os.path.join(self.project_folder, file)):
                debug_print(f"Warning: Missing suggested file '{file}' in the project folder.")

        # Check for the 'participants' folder
        if not os.path.isdir(self.participants_folder):
            raise FileNotFoundError("Error: The 'participants' folder is missing in the project folder.")

        # Check if there is at least one subfolder in 'participants', ideally named 'part-01'
        if not self.participant_folders:
            raise FileNotFoundError("Error: No participant subfolders found in the 'participants' folder.")
        if 'part-01' not in self.participant_folders:
            print("Warning: Ideally, participant folders should follow the naming convention 'part-x'.")

        # Iterate through participant subfolders
        for participant_folder in self.participant_folders:
            participant_path = os.path.join(self.participants_folder, participant_folder)

            # Check for session folders if project has sessions
            if self.multiple_sessions:
                session_folders = [f for f in os.listdir(participant_path) if
                                   os.path.isdir(os.path.join(participant_path, f))]
                if session_folders:
                    if 'ses-01' not in session_folders:
                        print(f"Warning: Ideally, the session folders should follow the naming convention 'ses-x'.")
                else:
                    print(f"Warning: No session folders found in '{participant_folder}'.")

            # Check for optional participant_metadata file
            metadata_file = os.path.join(participant_path, "participant_metadata")
            if not os.path.isfile(metadata_file):
                debug_print(f"Note: Optional 'participant_metadata' file is missing in '{participant_folder}'.")
                continue

            session_folders = participant_folder

            # Iterate through current level folders (participants or sessions)
            for session_folder in session_folders:
                session_path = os.path.join(participant_path, session_folder)
                task_folders = [f for f in os.listdir(session_path) if os.path.isdir(os.path.join(session_path, f))]

                # Check for tasks inside session folder
                for task_folder in task_folders:
                    task_path = os.path.join(session_path, task_folder)
                    task_files = [f for f in os.listdir(task_path) if os.path.isfile(os.path.join(task_path, f))]

                    # Check naming convention for files in the task folder
                    for file in task_files:
                        if self.multiple_sessions:
                            pattern = fr"^{participant_folder}_{session_folder}_{task_folder}.*"
                        else:
                            pattern = fr"^{participant_folder}_{task_folder}.*"
                        if not re.match(pattern, file):
                            debug_print(f"Warning: File '{file}' in '{task_folder}' does not follow the LPDS naming conventions")

    def derivative_dir_creator(self):
        # Create the 'derivatives' folder if it doesn't exist
        derivatives_folder = os.path.join(self.project_folder, "derivatives")
        if not os.path.exists(derivatives_folder):
            os.makedirs(derivatives_folder)