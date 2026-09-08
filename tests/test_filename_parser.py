from pathlib import Path

import pytest

from pelican_nlp.utils.filename_parser import parse_lpds_filename


def test_parse_fluency_filename():
    entities = parse_lpds_filename(
        "part-01_task-fluency_cat-semantic_acq-animals_text.txt"
    )
    assert entities["part"] == "01"
    assert entities["task"] == "fluency"
    assert entities["cat"] == "semantic"
    assert entities["acq"] == "animals"
    assert entities["suffix"] == "text"
    assert entities["extension"] == ".txt"


def test_parse_session_and_run():
    entities = parse_lpds_filename(
        "part-02_ses-01_task-interview_acq-schizophrenia_run-01_transcript.rtf"
    )
    assert entities["part"] == "02"
    assert entities["ses"] == "01"
    assert entities["run"] == "01"
    assert entities["suffix"] == "transcript"
    assert entities["extension"] == ".rtf"


def test_parse_path_uses_basename():
    entities = parse_lpds_filename(
        str(Path("participants/part-01/fluency/part-01_task-fluency_text.txt"))
    )
    assert entities["part"] == "01"
    assert entities["task"] == "fluency"
