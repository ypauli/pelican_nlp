from pathlib import Path

from pelican_nlp.preprocessing.LPDS import LPDS
from pelican_nlp.utils.setup_functions import (
    is_metadata_entry,
    participant_instantiator,
    path_contains_metadata,
)


def test_is_metadata_entry_names():
    assert is_metadata_entry("metadata")
    assert is_metadata_entry("metadata.csv")
    assert is_metadata_entry("participant_metadata")
    assert is_metadata_entry("participants.tsv")
    assert not is_metadata_entry("part-01")
    assert not is_metadata_entry("fluency")
    assert not is_metadata_entry("part-01_task-fluency_text.txt")


def test_path_contains_metadata():
    root = Path("/proj/participants/part-01")
    assert path_contains_metadata(root / "metadata" / "notes.txt", root)
    assert not path_contains_metadata(root / "fluency" / "part-01_task-fluency_text.txt", root)


def _fluency_config():
    return {
        "task_name": "fluency",
        "input_file": "text",
        "number_of_speakers": 1,
        "has_multiple_sections": False,
        "has_section_titles": False,
        "section_identification": None,
        "number_of_sections": None,
    }


def test_metadata_sidecars_do_not_become_participants(tmp_path):
    participants = tmp_path / "participants"
    part = participants / "part-01" / "fluency"
    part.mkdir(parents=True)
    (part / "part-01_task-fluency_acq-animals_text.txt").write_text(
        "Katze; Hund", encoding="utf-8"
    )

    (participants / "metadata").mkdir()
    (participants / "metadata" / "codes.csv").write_text("id,code\n", encoding="utf-8")
    (participants / "metadata.csv").write_text("id,age\n01,20\n", encoding="utf-8")
    (participants / "participants.tsv").write_text("participant_id\n01\n", encoding="utf-8")
    (participants / "part-01" / "participant_metadata").write_text(
        "age: 20\n", encoding="utf-8"
    )
    nested = participants / "part-01" / "metadata"
    nested.mkdir()
    (nested / "part-01_task-fluency_should-not-load.txt").write_text(
        "ignore me", encoding="utf-8"
    )

    loaded = participant_instantiator(_fluency_config(), tmp_path)
    assert [p.name for p in loaded] == ["part-01"]
    assert len(loaded[0].documents) == 1
    assert loaded[0].documents[0].name.endswith("acq-animals_text.txt")


def test_lpds_ignores_metadata_folder_at_participants_root(tmp_path):
    participants = tmp_path / "participants"
    (participants / "part-01" / "fluency").mkdir(parents=True)
    (participants / "metadata").mkdir()
    lpds = LPDS(tmp_path, multiple_sessions=False)
    assert lpds.participant_folders == ["part-01"]
    lpds.LPDS_checker()
