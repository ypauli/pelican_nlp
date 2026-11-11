import csv
from pathlib import Path
from types import SimpleNamespace

from pelican_nlp.utils.csv_functions import store_features_to_csv


def test_store_embeddings_and_logits(tmp_path):
    doc = SimpleNamespace(name="part-01_task-fluency_acq-animals_text.txt")
    embeddings = [("cat", [0.1, 0.2]), ("dog", [0.3, 0.4])]
    embedding_path = store_features_to_csv(embeddings, tmp_path, doc, metric="embeddings")
    assert Path(embedding_path).exists()
    relative = Path(embedding_path).relative_to(tmp_path)
    assert relative.parts[:3] == ("embeddings", "part-01", "task-fluency")
    assert relative.name == "part-01_task-fluency_acq-animals_embeddings.csv"

    with open(embedding_path, encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0][:4] == ["section_key", "part_index", "token_index", "Token"]
    assert rows[1][3] == "cat"
    assert rows[1][2] == "0"

    logits = [
        {
            "token": "hello",
            "logprob_actual": -1.2,
            "logprob_max": -0.1,
            "entropy": 0.5,
            "most_likely_token": "hello",
        }
    ]
    logits_path = store_features_to_csv(logits, tmp_path, doc, metric="logits")
    with open(logits_path, encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0][0] == "token"
    assert rows[1][0] == "hello"


def test_embedding_rerun_replaces_and_sections_share_one_header(tmp_path):
    from pelican_nlp.utils.csv_functions import reset_output_sessions

    doc = SimpleNamespace(name="part-01_task-fluency_text.txt")
    path = Path(store_features_to_csv(
        [("cat", [0.1])], tmp_path, doc, metric="embeddings", section_key="one", part_index=0
    ))
    path.write_text("stale\n", encoding="utf-8")
    reset_output_sessions()
    store_features_to_csv(
        [("cat", [0.2])], tmp_path, doc, metric="embeddings", section_key="one", part_index=0
    )
    store_features_to_csv(
        [("dog", [0.3])], tmp_path, doc, metric="embeddings", section_key="two", part_index=1
    )
    rows = list(csv.reader(path.open(encoding="utf-8")))
    assert rows[0][0] == "section_key"
    assert [row[3] for row in rows[1:]] == ["cat", "dog"]
    assert [row[0] for row in rows[1:]] == ["one", "two"]
    assert all(row[0] != "New Section" for row in rows)
    assert "stale" not in path.read_text(encoding="utf-8")


def test_empty_first_embedding_section_uses_the_later_width(tmp_path):
    from pelican_nlp.utils.csv_functions import reset_output_sessions

    reset_output_sessions()
    doc = SimpleNamespace(name="part-01_task-fluency_text.txt")
    path = Path(store_features_to_csv(
        [], tmp_path, doc, metric="embeddings", section_key="empty", part_index=0
    ))
    store_features_to_csv(
        [("cat", [0.1, 0.2])], tmp_path, doc, metric="embeddings", section_key="body", part_index=1
    )
    rows = list(csv.reader(path.open(encoding="utf-8")))
    assert rows[0] == ["section_key", "part_index", "token_index", "Token", "Dim_0", "Dim_1"]
    assert len(rows) == 2
    assert rows[1][3] == "cat"
    assert len(rows[1]) == 6


def test_new_run_after_reset_replaces_a_deleted_file(tmp_path):
    from pelican_nlp.utils.csv_functions import reset_output_sessions

    reset_output_sessions()
    doc = SimpleNamespace(name="part-01_task-fluency_text.txt")
    path = Path(store_features_to_csv(
        [("cat", [0.1])], tmp_path, doc, metric="embeddings", section_key="one"
    ))
    path.unlink()
    store_features_to_csv(
        [("dog", [0.2])], tmp_path, doc, metric="embeddings", section_key="two"
    )
    continued = list(csv.reader(path.open(encoding="utf-8")))
    assert continued[0][0] != "section_key"
    assert continued[0][3] == "dog"
    reset_output_sessions()
    store_features_to_csv(
        [("bird", [0.3])], tmp_path, doc, metric="embeddings", section_key="three"
    )
    rows = list(csv.reader(path.open(encoding="utf-8")))
    assert rows[0][0] == "section_key"
    assert [row[3] for row in rows[1:]] == ["bird"]


def test_pipeline_run_resets_output_sessions():
    import inspect

    from pelican_nlp.main import Pelican

    assert "reset_output_sessions()" in inspect.getsource(Pelican.run)


def test_store_collection_file_without_part_entity(tmp_path):
    doc = SimpleNamespace(name="hansel.txt", source_folder="stories")
    path = store_features_to_csv(
        [("once", [0.1])], tmp_path, doc, metric="embeddings"
    )
    assert Path(path) == tmp_path / "embeddings" / "stories" / "hansel_embeddings.csv"


def test_store_without_part_does_not_raise(tmp_path):
    doc = SimpleNamespace(name="hansel.txt")
    path = store_features_to_csv(
        [("once", [0.1])], tmp_path, doc, metric="embeddings"
    )
    assert Path(path) == tmp_path / "embeddings" / "unassigned" / "hansel_embeddings.csv"
