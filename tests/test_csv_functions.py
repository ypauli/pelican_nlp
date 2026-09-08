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
    assert rows[0][0] == "Token"
    assert rows[1][0] == "cat"

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
