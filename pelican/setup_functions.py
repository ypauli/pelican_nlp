import os
from pelican.preprocessing import Subject
from pelican.document import Document
import shutil
import yaml
import sys

def subject_instantiator(config):
    project_folder = config['PATH_TO_PROJECT_FOLDER']
    path_to_subjects = os.path.join(project_folder, 'subjects')
    print('Instantiating Subjects...')
    subjects = [Subject(subject) for subject in os.listdir(path_to_subjects)]

    # Identifying all subject files
    for subject in subjects:
        session_paths = _get_subject_sessions(subject, project_folder)
        for session_path in session_paths:
            file_path = os.path.join(session_path, config['task_name']) + '/'
            subject.documents.extend(_instantiate_documents(file_path, config))
        print(f'all identified subject documents for subject {subject.subjectID}: {subject.documents}')
        for document in subject.documents:
            print(f'file_path is: {document.file_path}')
            parts = document.file_path.split(os.sep)
            print(f'parts are: {parts}')
            subject_ID, session, task = parts[-4], parts[-3], parts[-2]
            document.results_path = os.path.join(project_folder, 'derivatives', config['metric_to_extract'], subject_ID, session, task)
            print(document.results_path)

    return subjects

def _get_subject_sessions(subject, project_path):
    session_dir = os.path.join(os.path.join(project_path, 'subjects'), subject.subjectID)
    session_paths = [
        os.path.join(session_dir, session)
        for session in os.listdir(session_dir)
        if os.path.isdir(os.path.join(session_dir, session))
    ]
    subject.numberOfSessions = len(session_paths)
    return session_paths

def _instantiate_documents(filepath, config):
    return [
        Document(
            filepath,
            file_name,
            fluency=config['fluency_task'],
            has_sections=config['has_multiple_sections'],
            section_identifier=config['section_identification'],
            number_of_sections=config['number_of_sections'],
            num_speakers=config['number_of_speakers'],
            has_section_titles=config['has_section_titles']
        )
        for file_name in os.listdir(filepath)
    ]

def _remove_previous_derivative_dir(output_directory):
    if os.path.isdir(output_directory):
        shutil.rmtree(output_directory)

def ignore_files(directory, files):
    return [f for f in files if os.path.isfile(os.path.join(directory, f))]

def _load_config(config_path):
    try:
        with open(config_path, 'r') as stream:
            return yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        sys.exit(f"Error loading configuration: {exc}")

