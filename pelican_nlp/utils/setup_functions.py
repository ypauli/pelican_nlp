import os
import shutil
import yaml
import sys
from pelican_nlp.core.subject import Subject
from .filename_parser import parse_lpds_filename
from ..config import debug_print


def subject_instantiator(config, project_folder):
    path_to_subjects = os.path.join(project_folder, 'subjects')
    print('Instantiating Subjects...')
    
    # Get all subject directories that match sub-* pattern
    subjects = [
        Subject(subject_dir) 
        for subject_dir in os.listdir(path_to_subjects)
    ]

    # Identifying all subject files
    for subject in subjects:
        # Get subject ID from directory name (e.g., 'sub-01' -> '01')
        subject.subjectID = subject.name.split('-')[1]
        
        # Find all files for this subject recursively
        subject_path = os.path.join(path_to_subjects, subject.name)
        all_files = []
        for root, _, files in os.walk(subject_path):
            all_files.extend([os.path.join(root, f) for f in files])
        
        # Filter files by task name from config
        task_files = []
        for file_path in all_files:
            filename = os.path.basename(file_path)
            entities = parse_lpds_filename(filename)
            if entities.get('task') == config['task_name']:
                task_files.append((file_path, filename))

        # Instantiate documents for matching files
        for file_path, filename in task_files:
            entities = parse_lpds_filename(filename)
            document = _instantiate_document(file_path, filename, entities, config)
            subject.documents.append(document)

        debug_print(f'all identified subject documents for subject {subject.subjectID}: {subject.documents}')
        
        # Set up results paths for each document
        for document in subject.documents:
            entities = parse_lpds_filename(document.name)
            
            # Build derivatives path based on entities
            derivatives_parts = [project_folder, 'derivatives']
            
            # Always include subject
            derivatives_parts.append(f"sub-{entities['sub']}")
            
            # Add session if present
            if 'ses' in entities:
                derivatives_parts.append(f"ses-{entities['ses']}")
            
            # Add task
            derivatives_parts.append(f"task-{entities['task']}")
            
            document.results_path = os.path.join(*derivatives_parts)

    return subjects

def _instantiate_document(filepath, filename, entities, config):
    """Create appropriate document instance based on config and entities"""

    common_kwargs = {
        'file_path': os.path.dirname(filepath),
        'name': filename,
        'subject_ID': entities.get('sub'),
        'task': entities.get('task'),
        # Check for specific entities that might indicate document type
        'fluency': 'cat' in entities and entities['cat'] == 'semantic',
        'num_speakers': config['number_of_speakers'],
    }

    if config['input_file'] == 'text':
        from pelican_nlp.core.document import Document
        return Document(
            **common_kwargs,
            # Use entities for section information if available, fall back to config
            has_sections=bool(entities.get('sections', config['has_multiple_sections'])),
            section_identifier=config['section_identification'],
            number_of_sections=config['number_of_sections'],
            has_section_titles=config['has_section_titles'],
            # Add any additional entities as attributes
            session=entities.get('ses'),
            acquisition=entities.get('acq'),
            category=entities.get('cat'),
            run=entities.get('run'),
        )
    elif config['input_file'] == 'audio':
        from pelican_nlp.core.audio_document import AudioFile
        return AudioFile(
            **common_kwargs,
            # Add audio-specific entities
            recording_type=entities.get('rec'),
            channel=entities.get('ch'),
            run=entities.get('run'),
        )

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
