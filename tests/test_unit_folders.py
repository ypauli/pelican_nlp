from pathlib import Path

from pelican_nlp.core.corpus import Corpus
from pelican_nlp.preprocessing.LPDS import LPDS
from pelican_nlp.utils.setup_functions import participant_instantiator


def _text_config(**overrides):
    config = {
        "task_name": "fluency",
        "input_file": "text",
        "number_of_speakers": 1,
        "has_multiple_sections": False,
        "has_section_titles": False,
        "section_identification": None,
        "number_of_sections": None,
    }
    config.update(overrides)
    return config


def test_collection_folder_loads_untagged_files_despite_task_name(tmp_path):
    stories = tmp_path / "participants" / "stories"
    stories.mkdir(parents=True)
    (stories / "hansel.txt").write_text("Once upon a time", encoding="utf-8")
    (stories / "hansel_and_gretel.txt").write_text("The woods", encoding="utf-8")
    (stories / "readme.txt").write_text("ignore me", encoding="utf-8")

    loaded = participant_instantiator(_text_config(), tmp_path)
    assert [unit.name for unit in loaded] == ["stories"]
    assert loaded[0].kind == "collection"
    assert loaded[0].participantID == "stories"
    names = sorted(document.name for document in loaded[0].documents)
    assert names == ["hansel.txt", "hansel_and_gretel.txt"]
    document = loaded[0].documents[0]
    assert document.source_folder == "stories"
    assert Path(document.results_path) == tmp_path / "derivatives" / "stories"


def test_part_folder_task_filter_and_results_path_unchanged(tmp_path):
    fluency = tmp_path / "participants" / "part-01" / "fluency"
    fluency.mkdir(parents=True)
    (fluency / "part-01_task-fluency_acq-animals_text.txt").write_text(
        "Katze; Hund", encoding="utf-8"
    )
    (fluency / "part-01_task-imgdesc_acq-beach_text.txt").write_text(
        "A beach", encoding="utf-8"
    )

    loaded = participant_instantiator(_text_config(), tmp_path)
    assert loaded[0].kind == "participant"
    assert loaded[0].participantID == "01"
    assert [document.name for document in loaded[0].documents] == [
        "part-01_task-fluency_acq-animals_text.txt"
    ]
    results = Path(loaded[0].documents[0].results_path)
    assert results == tmp_path / "derivatives" / "part-01" / "task-fluency"


def test_mixed_participant_and_collection_folders(tmp_path):
    part = tmp_path / "participants" / "part-01" / "fluency"
    stories = tmp_path / "participants" / "stories"
    part.mkdir(parents=True)
    stories.mkdir(parents=True)
    (part / "part-01_task-fluency_acq-animals_text.txt").write_text(
        "Katze", encoding="utf-8"
    )
    (stories / "hansel.txt").write_text("Once upon a time", encoding="utf-8")

    loaded = participant_instantiator(_text_config(), tmp_path)
    by_name = {unit.name: unit for unit in loaded}
    assert set(by_name) == {"part-01", "stories"}
    assert len(by_name["part-01"].documents) == 1
    assert len(by_name["stories"].documents) == 1


def test_corpus_accepts_folder_name_without_hyphen(tmp_path):
    corpus = Corpus("stories", [], {"task_name": None}, tmp_path)
    assert corpus.name == "stories"
    assert corpus.key == "unit"
    assert corpus.value == "stories"


def test_lpds_collection_only_does_not_warn_about_part_x(tmp_path, capsys):
    (tmp_path / "participants" / "stories").mkdir(parents=True)
    lpds = LPDS(tmp_path, multiple_sessions=False)
    assert lpds.participant_folders == ["stories"]
    lpds.LPDS_checker()
    assert "part-x" not in capsys.readouterr().out
