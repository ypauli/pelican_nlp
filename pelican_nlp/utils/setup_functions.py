import os
from pelican_nlp.core.subject import Subject
import shutil
import yaml
import sys

def subject_instantiator(config, project_folder):
    path_to_subjects = os.path.join(project_folder, 'subjects')
    print('Instantiating Subjects...')
    subjects = [Subject(subject) for subject in os.listdir(path_to_subjects)]

    # Identifying all subject files
    for subject in subjects:
        if config['multiple_sessions']:
            paths = _get_subject_sessions(subject, project_folder)
        else:
            paths = [os.path.join(path_to_subjects, subject.subjectID)]

        for path in paths:
            file_path = os.path.join(path, config['task_name'])
            subject.documents.extend(_instantiate_documents(file_path, subject.subjectID, config))
        print(f'all identified subject documents for subject {subject.subjectID}: {subject.documents}')
        for document in subject.documents:
            parts = document.file_path.split(os.sep)
            
            # Adjust path components based on whether session exists
            if config.get('multiple_sessions', False):
                subject_ID, session, task = parts[-4], parts[-3], parts[-2]
                document.results_path = os.path.join(project_folder, 'derivatives', subject_ID, session, task)
            else:
                subject_ID, task = parts[-3], parts[-2]
                document.results_path = os.path.join(project_folder, 'derivatives', subject_ID, task)

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

def _instantiate_documents(filepath, subject, config):

    if config['input_file']=='text':
        from pelican_nlp.core.document import Document
        return [
            Document(
                filepath,
                file_name,
                subject_ID = subject,
                task=config['task_name'],
                fluency=config['fluency_task'],
                has_sections=config['has_multiple_sections'],
                section_identifier=config['section_identification'],
                number_of_sections=config['number_of_sections'],
                num_speakers=config['number_of_speakers'],
                has_section_titles=config['has_section_titles']
            )
            for file_name in os.listdir(filepath)
        ]

    elif config['input_file']=='audio':
        from pelican_nlp.core.audio_document import AudioFile
        return [
            AudioFile(
                filepath,
                file_name,
                subject_ID=subject,
                task=config['task_name'],
                fluency=config['fluency_task'],
                num_speakers=config['number_of_speakers'],
            )
            for file_name in os.listdir(filepath)
        ]

def remove_previous_derivative_dir(output_directory):
    if os.path.isdir(output_directory):
        shutil.rmtree(output_directory)

def ignore_files(directory, files):
    return [f for f in files if os.path.isfile(os.path.join(directory, f))]

def load_config(config_path):
    try:
        with open(config_path, 'r') as stream:
            return yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        sys.exit(f"Error loading configuration: {exc}")
