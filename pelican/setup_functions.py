import os
from pelican.preprocessing import Subject
from pelican.document import Document
import shutil
import yaml
import sys

def subject_instatiator(config):
    path_to_subjects = os.path.join(config['PATH_TO_PROJECT_FOLDER'], 'Subjects')
    print('Instantiating Subjects...')
    subjects = [Subject(subject) for subject in os.listdir(path_to_subjects)]

    # Identify sessions per subject
    if config['multiple_sessions']:
        for subject in subjects:
            subject.numberOfSessions = len(os.listdir(os.path.join(path_to_subjects, str(subject.subjectID))))

    return subjects

def _create_documents(filepath, corpus_name, config):
    return [
        Document(
            filepath,
            file_name,
            corpus_name,
            has_sections=config['has_multiple_sections'],
            section_identifier=config['section_identification'],
            number_of_sections=config['number_of_sections']
        )
        for file_name in os.listdir(filepath)
    ]

def _reset_output_directory(output_directory, source_dir):
    if os.path.isdir(output_directory):
        shutil.rmtree(output_directory)
    shutil.copytree(source_dir, output_directory, ignore=ignore_files)
    os.mkdir(os.path.join(output_directory, 'results_consolidation'))

def ignore_files(directory, files):
    return [f for f in files if os.path.isfile(os.path.join(directory, f))]

def _load_config(config_path):
    try:
        with open(config_path, 'r') as stream:
            return yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        sys.exit(f"Error loading configuration: {exc}")

