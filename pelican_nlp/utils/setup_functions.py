import os
import shutil
import yaml
import sys
from pathlib import Path
from pelican_nlp.core.participant import Participant
from .filename_parser import parse_lpds_filename
from .lpds_paths import derivatives_subdir, file_matches_task, is_data_file, unit_id_from_folder
from pelican_nlp.config import debug_print
from pelican_nlp.config_defaults import apply_config_defaults


def is_hidden_or_system_file(filename):
    """True for hidden/system files that should not be treated as subject data."""
    if not filename:
        return False
    if filename.startswith("."):
        return True
    return filename in {"Thumbs.db", "desktop.ini"}


def resolve_project_config(project) -> Path:
    """Return the YAML for a project folder or a YAML file path."""
    path = Path(project).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in {".yml", ".yaml"}:
            raise ValueError(f"Not a YAML configuration file: {path}")
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Project path does not exist: {path}")
    names = [
        name
        for name in os.listdir(path)
        if name.endswith((".yml", ".yaml")) and not is_hidden_or_system_file(name)
    ]
    if not names:
        raise FileNotFoundError(
            f"No .yml or .yaml configuration file found in {path}."
        )
    if len(names) > 1:
        raise ValueError(
            "Multiple configuration files found. "
            "Please ensure only one configuration file is present."
        )
    return path / names[0]


# Sidecars that may sit in or under participants/ without being subject data.
_METADATA_STEMS = {"metadata", "participant_metadata"}
_METADATA_FILENAMES = {
    "participants.tsv",
    "participants.csv",
    "participants.json",
}


def is_metadata_entry(name):
    """Return True for a metadata file or folder name (not a participant or data file)."""
    base = os.path.basename(str(name).rstrip("/\\"))
    if not base:
        return False
    lower = base.lower()
    if lower in _METADATA_FILENAMES:
        return True
    stem, _ext = os.path.splitext(lower)
    return stem in _METADATA_STEMS


def path_contains_metadata(path, root=None):
    """Return True if any path component under ``root`` is a metadata sidecar."""
    if root:
        try:
            relative = os.path.relpath(path, root)
        except ValueError:
            relative = path
    else:
        relative = path
    return any(is_metadata_entry(part) for part in Path(relative).parts)


def participant_instantiator(config, project_folder):
    path_to_participants = os.path.join(project_folder, 'participants')
    
    # Only subject folders. Skip files and metadata sidecars at this level.
    participants = []
    for entry in os.listdir(path_to_participants):
        if is_hidden_or_system_file(entry) or is_metadata_entry(entry):
            continue
        entry_path = os.path.join(path_to_participants, entry)
        if not os.path.isdir(entry_path):
            continue
        participants.append(Participant(entry))

    # Identifying all files in each unit folder
    for participant in participants:
        participant_path = os.path.join(path_to_participants, participant.name)
        all_files = []
        for root, dirs, files in os.walk(participant_path):
            dirs[:] = [
                d for d in dirs
                if not is_hidden_or_system_file(d) and not is_metadata_entry(d)
            ]
            if path_contains_metadata(root, participant_path):
                continue
            filtered_files = [
                f for f in files
                if not is_hidden_or_system_file(f) and not is_metadata_entry(f)
            ]
            all_files.extend([os.path.join(root, f) for f in filtered_files])

        task_name = config.get('task_name')
        for file_path in all_files:
            filename = os.path.basename(file_path)
            if not is_data_file(filename):
                continue
            entities = parse_lpds_filename(filename)
            if not file_matches_task(entities, task_name):
                continue
            document = _instantiate_document(
                file_path,
                filename,
                entities,
                config,
                source_folder=participant.name,
                unit_kind=participant.kind,
            )
            document.lpds_entities = entities
            document.results_path = os.path.join(
                project_folder,
                'derivatives',
                str(derivatives_subdir(participant.name, filename, entities)),
            )
            participant.documents.append(document)

        debug_print(
            f'all identified documents for {participant.name} '
            f'({participant.kind}): {participant.documents}'
        )

    return participants

def _instantiate_document(filepath, filename, entities, config, source_folder=None, unit_kind=None):
    """Create appropriate document instance based on config and entities"""

    participant_id = entities.get('part')
    if participant_id is None and unit_kind == 'participant' and source_folder:
        participant_id = unit_id_from_folder(source_folder)

    common_kwargs = {
        'file_path': os.path.dirname(filepath),
        'name': filename,
        'participant_ID': participant_id,
        'source_folder': source_folder,
        'unit_kind': unit_kind,
        'task': entities.get('task'),
        'num_speakers': config.get('number_of_speakers', 1),
    }

    if config['input_file'] == 'text':
        from pelican_nlp.core.document import Document
        return Document(
            **common_kwargs,
            # Use entities for section information if available, fall back to config
            has_sections=bool(entities.get('sections', config.get('has_multiple_sections', False))),
            section_identifier=config.get('section_identification'),
            number_of_sections=config.get('number_of_sections'),
            has_section_titles=config.get('has_section_titles', False),
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

def load_config(config_path):
    try:
        with open(config_path, 'r') as stream:
            loaded = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        sys.exit(f"Error loading configuration: {exc}")
    return apply_config_defaults(loaded or {})
