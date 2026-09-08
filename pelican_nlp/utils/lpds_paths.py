"""LPDS identity and derivatives path helpers.

Unit folders under ``participants/`` are either participants (``part-*``)
or collections (any other directory name, e.g. ``stories``). Output paths
always start with that folder name, then optional filename tags in a
fixed order. Missing tags are omitted so existing ``part-*/task-*``
layouts stay unchanged.
"""

from pathlib import Path

from .filename_parser import parse_lpds_filename

# Directory tags after the unit folder. Matches the historical writer
# (part / ses / task only). Other filename entities stay in the CSV name.
PATH_ENTITY_ORDER = ("ses", "task")

DATA_EXTENSIONS = {
    ".txt",
    ".docx",
    ".rtf",
    ".pdf",
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
}

_SKIP_STEMS = {"readme", "notes", "license", "changelog"}

UNIT_KIND_PARTICIPANT = "participant"
UNIT_KIND_COLLECTION = "collection"


def is_participant_folder(name):
    """True when a unit folder follows the ``part-<id>`` convention."""
    base = Path(str(name)).name
    if not base.lower().startswith("part-"):
        return False
    return len(base) > 5


def unit_kind(folder_name):
    return UNIT_KIND_PARTICIPANT if is_participant_folder(folder_name) else UNIT_KIND_COLLECTION


def unit_id_from_folder(folder_name):
    """Participant id (``01``) or the collection folder name as-is."""
    name = Path(str(folder_name)).name
    if is_participant_folder(name):
        return name.split("-", 1)[1]
    return name


def is_data_file(filename):
    """True for input files the pipeline can load (not notes/readme)."""
    path = Path(filename)
    if path.stem.lower() in _SKIP_STEMS:
        return False
    return path.suffix.lower() in DATA_EXTENSIONS


def file_matches_task(entities, task_name):
    """Keep the file if it has no task tag, or the tag matches ``task_name``.

    When ``task_name`` is unset, every data file is eligible.
    """
    if not task_name:
        return True
    file_task = entities.get("task")
    if file_task is None:
        return True
    return str(file_task) == str(task_name)


def resolve_unit_folder(unit_folder, entities=None):
    """Folder label for derivatives. Falls back to ``part-<id>`` from the filename."""
    if unit_folder:
        return str(unit_folder)
    entities = entities or {}
    if "part" in entities and entities["part"] is not None:
        return f"part-{entities['part']}"
    return "unassigned"


def csv_output_filename(filename, metric, entities=None):
    """``{stem}_{metric}.csv``, stripping a trailing LPDS suffix such as ``_text``."""
    entities = entities if entities is not None else parse_lpds_filename(filename)
    base_filename = Path(filename).stem
    has_lpds_keys = any(key not in ("extension", "suffix") for key in entities)
    suffix = entities.get("suffix")
    if has_lpds_keys and suffix:
        base_filename = base_filename.replace(f"_{suffix}", "")
    return f"{base_filename}_{metric}.csv"


def derivatives_subdir(unit_folder, filename, entities=None):
    """Relative directory under a metric folder (no metric prefix).

    Examples::

        part-01 + part-01_task-fluency_acq-animals_text.txt
            -> part-01/task-fluency
        stories + hansel.txt
            -> stories
    """
    entities = entities if entities is not None else parse_lpds_filename(filename)
    unit = resolve_unit_folder(unit_folder, entities)
    segments = [unit]
    for key in PATH_ENTITY_ORDER:
        if key not in entities or entities[key] is None:
            continue
        segment = f"{key}-{entities[key]}"
        if segment == unit:
            continue
        segments.append(segment)
    return Path(*segments)


def derivatives_output_path(derivatives_dir, unit_folder, filename, metric, entities=None):
    """Absolute CSV path: ``{derivatives}/{metric}/{unit}/[ses]/[task]/{stem}_{metric}.csv``."""
    entities = entities if entities is not None else parse_lpds_filename(filename)
    relative = Path(str(metric)) / derivatives_subdir(unit_folder, filename, entities)
    directory = Path(derivatives_dir) / relative
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory / csv_output_filename(filename, metric, entities))


def aggregation_unit_key(file_path, derivatives_dir):
    """Unit folder from a derivatives path (``…/{metric}/{unit}/…``)."""
    try:
        relative = Path(file_path).resolve().relative_to(Path(derivatives_dir).resolve())
    except ValueError:
        relative = Path(file_path)
    parts = relative.parts
    if len(parts) >= 2:
        return parts[1]
    return Path(file_path).name.split("_")[0]


def document_entities(document):
    cached = getattr(document, "lpds_entities", None)
    if cached is not None:
        return cached
    return parse_lpds_filename(getattr(document, "name", "") or "")


def unit_folder_for_document(document):
    """Unit folder stored on the document, else ``part-<id>`` from its filename."""
    return resolve_unit_folder(
        getattr(document, "source_folder", None),
        document_entities(document),
    )


def documents_matching_entity(documents, entity):
    """Return documents whose filename entities match ``key-value`` or a bare value."""
    matched = []
    if "-" in entity:
        key, value = entity.split("-", 1)
        for document in documents:
            entities = document_entities(document)
            if key in entities and str(entities[key]) == value:
                matched.append(document)
    else:
        for document in documents:
            entities = document_entities(document)
            if any(str(val) == entity for val in entities.values()):
                matched.append(document)
    return matched


def _grouped_jobs(documents, corpus_key, corpus_values, leftover_key):
    assigned = set()
    values = list(corpus_values or [])
    documents = list(documents)
    if corpus_key:
        for value in values:
            entity = f"{corpus_key}-{value}"
            matched = documents_matching_entity(documents, entity)
            if not matched:
                continue
            for document in matched:
                assigned.add(id(document))
            yield entity, matched

    leftovers = {}
    for document in documents:
        if id(document) in assigned:
            continue
        entities = document_entities(document)
        if corpus_key and corpus_key in entities:
            continue
        leftovers.setdefault(leftover_key(document), []).append(document)
    for name, matched in leftovers.items():
        if matched:
            yield name, matched


def grouped_corpus_jobs(participants, corpus_key=None, corpus_values=None):
    """Yield ``(corpus_name, documents)`` for configured groups, then leftover units."""
    unit_of = {}
    documents = []
    for unit in participants:
        for document in unit.documents:
            documents.append(document)
            unit_of[id(document)] = unit.name
    yield from _grouped_jobs(
        documents,
        corpus_key,
        corpus_values,
        leftover_key=lambda document: unit_of.get(id(document))
        or unit_folder_for_document(document),
    )


def grouped_document_jobs(documents, corpus_key=None, corpus_values=None):
    """Same grouping as ``grouped_corpus_jobs`` for a flat document list."""
    yield from _grouped_jobs(
        documents, corpus_key, corpus_values, leftover_key=unit_folder_for_document
    )
