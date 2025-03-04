import re
import os

class LPDS:
    def __init__(self, project_folder):
        self.project_folder = project_folder
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
                print(f"Warning: Missing suggested file '{file}' in the project folder.")

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

            # Check for session folders
            session_folders = [f for f in os.listdir(subject_path) if
                               os.path.isdir(os.path.join(subject_path, f))]
            if not session_folders:
                print(f"Warning: No session folders found in '{subject_folder}'.")
            if 'ses-01' not in session_folders:
                print(f"Warning: Ideally, there session folders should follow the naming convention 'ses-x'.")

            # Check for optional subject_metadata file
            metadata_file = os.path.join(subject_path, "subject_metadata")
            if not os.path.isfile(metadata_file):
                print(f"Note: Optional 'subject_metadata' file is missing in '{subject_folder}'.")

            # Iterate through session folders
            for session_folder in session_folders:
                session_path = os.path.join(subject_path, session_folder)
                task_folders = [f for f in os.listdir(session_path) if os.path.isdir(os.path.join(session_path, f))]

                # Check for tasks inside session folder
                for task_folder in task_folders:
                    task_path = os.path.join(session_path, task_folder)
                    task_files = [f for f in os.listdir(task_path) if os.path.isfile(os.path.join(task_path, f))]

                    # Check naming convention for files in the task folder
                    for file in task_files:
                        pattern = fr"^{subject_folder}_{session_folder}_{task_folder}.*"
                        if not re.match(pattern, file):
                            print(f"Warning: File '{file}' in '{task_folder}' does not follow the LPDS naming conventions")

    def derivative_dir_creator(self, metric):
        # Create the 'derivatives' folder if it doesn't exist
        derivatives_folder = os.path.join(self.project_folder, "derivatives")
        if not os.path.exists(derivatives_folder):
            os.makedirs(derivatives_folder)

        # Iterate through specified derivative names
        derivative_path = os.path.join(derivatives_folder, metric)

        # Check if the derivative folder already exists
        if os.path.exists(derivative_path):
            print(f"Warning: Derivative folder '{metric}' already exists. Exiting to not overwrite previous results.")
            return

        # Create the derivative subfolder
        os.makedirs(derivative_path)

        for subject_folder in self.subject_folders:
            subject_path = os.path.join(self.subjects_folder, subject_folder)
            new_subject_path = os.path.join(derivative_path, subject_folder)
            os.makedirs(new_subject_path)

            session_folders = [f for f in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, f))]

            for session_folder in session_folders:
                session_path = os.path.join(subject_path, session_folder)
                new_session_path = os.path.join(new_subject_path, session_folder)
                os.makedirs(new_session_path)

                task_folders = [f for f in os.listdir(session_path) if os.path.isdir(os.path.join(session_path, f))]

                for task_folder in task_folders:
                    task_path = os.path.join(session_path, task_folder)
                    new_task_path = os.path.join(new_session_path, task_folder)
                    os.makedirs(new_task_path)