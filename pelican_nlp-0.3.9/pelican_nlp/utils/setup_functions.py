import os
import shutil
import yaml
import sys
from pelican_nlp.core.participant import Participant
from .filename_parser import parse_lpds_filename
from pelican_nlp.config import debug_print


def participant_instantiator(config, project_folder):
    path_to_participants = os.path.join(project_folder, 'participants')
    print('Instantiating Participants...')
    
    # Get all participant directories that match part-* pattern
    participants = [
        Participant(participant_dir) 
        for participant_dir in os.listdir(path_to_participants)
    ]

    # Identifying all participant files
    for participant in participants:
        # Get participant ID from directory name (e.g., 'part-01' -> '01')
        participant.participantID = participant.name.split('-')[1]
        
        # Find all files for this participant recursively
        participant_path = os.path.join(path_to_participants, participant.name)
        all_files = []
        for root, _, files in os.walk(participant_path):
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
            participant.documents.append(document)

        debug_print(f'all identified participant documents for participant {participant.participantID}: {participant.documents}')
        
        # Set up results paths for each document
        for document in participant.documents:
            entities = parse_lpds_filename(document.name)
            
            # Build derivatives path based on entities
            derivatives_parts = [project_folder, 'derivatives']
            
            # Always include participant
            derivatives_parts.append(f"part-{entities['part']}")
            
            # Add session if present
            if 'ses' in entities:
                derivatives_parts.append(f"ses-{entities['ses']}")
            
            # Add task
            derivatives_parts.append(f"task-{entities['task']}")
            
            document.results_path = os.path.join(*derivatives_parts)

    return participants

def _instantiate_document(filepath, filename, entities, config):
    """Create appropriate document instance based on config and entities"""

    common_kwargs = {
        'file_path': os.path.dirname(filepath),
        'name': filename,
        'participant_ID': entities.get('part'),
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
